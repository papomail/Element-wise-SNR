import argparse
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import sqlite3
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image as ReportImage
from reportlab.lib.units import inch


# Setup logging
logging.basicConfig(level=logging.INFO)


def compute_snr(img_slice: np.ndarray) -> (int, tuple, tuple):
    """Compute the Signal-to-Noise Ratio (SNR) for a given image slice."""
    sorted_pixels_slice = np.sort(img_slice.flatten())
    non_zero_pixels = sorted_pixels_slice[sorted_pixels_slice > 0]
    num_non_zero_pixels = non_zero_pixels.size
    percentile = 0.05
    num_percentile_slice = int(percentile * num_non_zero_pixels)
    
    bright_threshold_slice = np.mean(non_zero_pixels[-num_percentile_slice-2 : -num_percentile_slice+2])
    dark_threshold_slice = np.mean(non_zero_pixels[num_percentile_slice-2 : num_percentile_slice+2])
    
    avg_brightest_slice = np.mean(sorted_pixels_slice[-num_percentile_slice:])
    avg_darkest_slice = np.mean(non_zero_pixels[:num_percentile_slice])
    
    # snr_slice = int(avg_brightest_slice / avg_darkest_slice)
    snr_slice = int(np.round(avg_brightest_slice / avg_darkest_slice))
    bright_indices_slice = np.where(img_slice > bright_threshold_slice)
    dark_indices_slice = np.where((img_slice <= dark_threshold_slice) & (img_slice > 0))
    
    return snr_slice, bright_indices_slice, dark_indices_slice


def retrieve_nested_dicom_value(ds: pydicom.FileDataset, tags: list) -> str:
    """Retrieve nested DICOM values given a sequence of tags."""
    for tag in tags:
        if tag in ds:
            if isinstance(ds[tag].value, pydicom.sequence.Sequence):
                ds = ds[tag].value[0]
            else:
                return ds[tag].value if tag == tags[-1] else None
        else:
            return None


def retrieve_all_nested_dicom_values(ds: pydicom.FileDataset, tags: list) -> list:
    """Retrieve all nested DICOM values for a given set of tags."""
    def get_values(ds, remaining_tags):
        current_tag = remaining_tags[0]
        if current_tag not in ds:
            return None
        if len(remaining_tags) == 1:
            if isinstance(ds[current_tag].value, pydicom.sequence.Sequence):
                return [item.value for item in ds[current_tag].value]
            else:
                return [ds[current_tag].value]
        if isinstance(ds[current_tag].value, pydicom.sequence.Sequence):
            results = []
            for item in ds[current_tag].value:
                nested_values = get_values(item, remaining_tags[1:])
                if nested_values:
                    results.extend(nested_values)
            return results
        else:
            return get_values(ds[current_tag].value, remaining_tags[1:])
    
    return get_values(ds, tags)


def extract_dicom_tags(dicom_data: pydicom.FileDataset) -> dict:
    """Extract relevant DICOM tags from the provided data."""
    try:
        coil = dicom_data.get('SharedFunctionalGroupsSequence')[0].get('MRReceiveCoilSequence')[0].get('ReceiveCoilName')
        element = retrieve_all_nested_dicom_values(dicom_data, [(0x5200, 0x9230), (0x0021, 0x11fe), (0x0021, 0x114f)])
    except Exception:
        coil = dicom_data.get('ReceiveCoilName', "N/A")
        element = retrieve_all_nested_dicom_values(dicom_data, [(0x0021, 0x114f)])
         
    return {
        "Institution": dicom_data.get('InstitutionName', "N/A"),
        "Scanner": retrieve_nested_dicom_value(dicom_data, [(0x0008, 0x1090)]),
        "ID": dicom_data.get('DeviceSerialNumber', "N/A"),
        "Date": retrieve_nested_dicom_value(dicom_data, [(0x0008, 0x0020)]),
        "Coil": coil,
        "Element": element,
    }



def create_overlay_img(img_slice: np.ndarray, bright_indices: tuple, dark_indices: tuple) -> np.ndarray:
    """Create an overlay image highlighting specific regions."""
    img_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
    color_img = np.stack([img_normalized] * 3, axis=-1)
    color_img[bright_indices] = [1, 0, 0]
    color_img[dark_indices] = [0, 0, 1]
    return color_img


def save_overlay(img_slice: np.ndarray, bright_indices: tuple, dark_indices: tuple, snr: int, scanner_id: str, coil_element: list, output_dir: str, scan_date: str, slice_num: int = None) -> None:
    """Save an overlay image highlighting specific regions."""
    color_img = create_overlay_img(img_slice, bright_indices, dark_indices)
    plt.imshow(color_img)
    plt.axis('off')
    plt.text(10, 15, f"Element: {coil_element[0]}", color='white', fontsize=12)
    plt.text(10, 27, f"SNR: {snr}", color='white', fontsize=12)
    plt.text(10, 39, f"Scan Date: {scan_date}", color='white', fontsize=12)
    filename = f"ID_{scanner_id}_Element_{coil_element[0]}_SNR_{snr}_Slice_{slice_num}.png" if slice_num else f"ID_{scanner_id}_Element_{coil_element[0]}_SNR_{snr}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def process_dicom_data(results: list, dicom_data: pydicom.FileDataset, filename: str, output_dir: str, serial: str) -> None:
    """Process DICOM data: extract relevant data and save overlay images."""
    if dicom_data.pixel_array.shape[0] < 50:
        for i, img_slice in enumerate(dicom_data.pixel_array):
            tags = extract_dicom_tags(dicom_data)
            snr, bright_indices, dark_indices = compute_snr(img_slice)
            save_overlay(img_slice, bright_indices, dark_indices, snr, tags["ID"], tags["Element"], output_dir, tags['Date'], i)
            tags["SNR"] = snr
            tags["File"] = filename
            tags["SerialNumber"] = serial
            results.append(tags)
    else:
        tags = extract_dicom_tags(dicom_data)
        snr, bright_indices, dark_indices = compute_snr(dicom_data.pixel_array)
        save_overlay(dicom_data.pixel_array, bright_indices, dark_indices, snr, tags["ID"], tags["Element"], output_dir, tags['Date'])
        tags["SNR"] = snr
        tags["File"] = filename
        tags["SerialNumber"] = serial
        results.append(tags)


def analyze_dicom_files(mypath: str, serial: str = '') -> pd.DataFrame:
    """Analyze DICOM files in a directory and extract relevant information."""
    results = []
    output_dir = os.path.join(mypath, 'SNR_test_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for n, filename in enumerate(Path(mypath).rglob('*')):
        if filename.suffix in ['.dcm', ''] and filename.stem != 'DICOMDIR' and filename.is_file():
            try:
                dicom_data = pydicom.dcmread(filename)
            except Exception as e:
                # print(f"Ignoring file {n} {filename}: {e}")
                print(f"Ignoring file {n} {filename}")
                continue

            process_dicom_data(results, dicom_data, filename, output_dir, serial)
            
    df = pd.DataFrame(results)
    df['Element'] = df['Element'].apply(lambda x: x[0] if isinstance(x, list) else x)
    df = df[df.groupby('Element')['SNR'].transform(max) == df['SNR']]
    df['SNR'] = df['SNR'].astype(int)
    df = df[~df['Element'].str.contains('-')]
    df = df[~df['Element'].str.contains(',')]
    df = df.sort_values(by="Element")
    df["SerialNumber"] = serial
    csv_path = os.path.join(output_dir, 'SNR_test_results.csv')
    df.to_csv(csv_path, index=False)
    return df        


def create_snr_chart(df: pd.DataFrame, threshold_value: int = 50) -> str:
    """Create a bar chart visualizing the SNR values for different elements."""
    plt.figure(figsize=(10, 6))
    colors = ['red' if snr < threshold_value else 'orange' if snr < 2*threshold_value else 'cadetblue' for snr in df['SNR']]
    plt.bar(df['Element'], df['SNR'], color=colors)
    plt.axhline(y=50, color='gray', linestyle='--')
    plt.xlabel('Element')
    plt.ylabel('SNR')
    plt.title('SNR per Element')
    
    if len(df['Element']) > 30:
        plt.xticks(rotation=90, ha="right")
    
    image_path = 'snr_chart.png'
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return image_path



def generate_pdf_report_reportlab(df: pd.DataFrame, image_path: str, scan_date: str, coil_name: str, scanner_model: str, collage_path: str, out_path: str = '', threshold_value: int = 50) -> str:
    """Generate a PDF report using the ReportLab library."""
    # Create a new document with a specified pagesize
    report_path =f'{out_path}/Coil_Test_Report.pdf'
    doc = SimpleDocTemplate(report_path, pagesize=A4)

    # Create an empty list to hold the content
    elements = []

    # Title
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    title = Paragraph("Coil Test Report", title_style)
    elements.append(title)

    # Add some space
    elements.append(Paragraph("<br/><br/>", styles['Normal']))

    # Scan Information Table
    scan_info = [["Coil Name", coil_name],
                 ["Scan Date", scan_date],
                 ["Institution", df['Institution'].iloc[0]],
                 ["Scanner Model", scanner_model]]

    table = Table(scan_info, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkcyan),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.ivory),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)

    # Add some space
    elements.append(Paragraph("<br/><br/>", styles['Normal']))

    # SNR Chart
    snr_image = ReportImage(image_path, width=500, height=300)
    elements.append(snr_image)

    # Add some space
    elements.append(Paragraph("<br/><br/>", styles['Normal']))

    # SNR Table
    snr_data = [["Element", "SNR"]]
    for _, row in df.iterrows():
        snr_data.append([row['Element'], str(row['SNR'])])
    snr_table = Table(snr_data, colWidths=[2*inch, 2*inch])

    # Define the table style
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkcyan),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.ivory),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]

    # Check SNR values and set the background color of the cell to red if below the threshold
    for i, data_row in enumerate(snr_data[1:], 1):
        if int(data_row[1]) < threshold_value:  
            table_style.append(('BACKGROUND', (1, i), (1, i), colors.red))
        elif int(data_row[1]) < 2*threshold_value:  
            table_style.append(('BACKGROUND', (1, i), (1, i), colors.orange))
    snr_table.setStyle(TableStyle(table_style))
    elements.append(snr_table)

    # Add some space
    elements.append(Paragraph("<br/><br/>", styles['Normal']))

    # Collage Image
    max_width = 7 * inch
    max_height = 9 * inch

    # Open the collage image with PIL to get its size
    with Image.open(collage_path) as img:
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height

        # Calculate new image dimensions
        new_width = max_width
        new_height = max_width / aspect_ratio
        if new_height > max_height:
            new_height = max_height
            new_width = max_height * aspect_ratio

        collage_image = ReportImage(str(collage_path), width=new_width, height=new_height)
        elements.append(collage_image)

    # Collage caption
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Italic'],
        fontSize=10,
        leading=12,
        alignment=1,  # Center aligned
    )
    caption = Paragraph('Figure: 5% high & low intensity voxels for element-wise SNR calculation', caption_style)
    elements.append(caption)

    # Generate the PDF
    doc.build(elements)

    return report_path


def reduce_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the DataFrame to only include unique and relevant information."""
    # Create the elements_snr dictionary
    elements_snr = dict(zip(df['Element'], df['SNR']))

    # Drop the Element, SNR and File, columns
    df_dropped = df.drop(columns=['Element', 'SNR', 'File'])

    # Drop duplicate rows
    df_dropped = df_dropped.drop_duplicates().reset_index(drop=True)

    # Add the elements_snr dictionary as a new column
    df_dropped['Element_SNR'] = [elements_snr]

    return  df_dropped #one-line df

            
def clean_unnecessary_images(df: pd.DataFrame, out_folder: str) -> None:
    """Remove unnecessary images from the output folder.
    Delete unnecessary jpg images from the output folder. Only keeps images whose filenames match the 'Element' and 'SNR' 
    values from the provided dataframe.
    
    Args:
    - df (pandas.DataFrame): The dataframe containing 'Element' and 'SNR' values.
    - out_folder (str): Path to the output folder containing the images.
    """
    # Generate a list of necessary filenames based on the df
    necessary_files = [f"ID_{row['ID']}_Element_{row['Element']}_SNR_{row['SNR']}" for _, row in df.iterrows()]
    # Check every file in the out_folder
    for filename in os.listdir(out_folder):
        base_name, ext = os.path.splitext(filename)
        added_text = '_Slice_x'
        if added_text[0:-1] in base_name:
            base_name=base_name[0:-len(added_text)]
        if ext == '.png' and base_name not in necessary_files:            
            os.remove(os.path.join(out_folder, filename))     
            


def create_collage(out_folder: str, output_collage_filename: str = "collage.png") -> None:
    """Create a collage image from individual images in the output folder."""
    # Get all .png files in the out_folder
    filenames = [f for f in sorted(Path(out_folder).rglob("ID*.png"))]
    print(f'{len(filenames)=}')
    # Determine the size of an individual image
    with Image.open(filenames[0]) as img:
        img_width, img_height = img.size
    
    # Compute rows and columns for collage
    num_images = len(filenames)
    aspect_ratio = 0.8
    collage_width = int((num_images * aspect_ratio)**0.5)
    collage_height = int(num_images / collage_width)
    
    # Adjust if there are more rows than needed
    while collage_width * collage_height < num_images:
        collage_height += 1

    # Create blank canvas for the collage
    collage = Image.new('RGB', (img_width * collage_width, img_height * collage_height))
    
    # Paste images onto canvas one by one
    for i, filename in enumerate(filenames):
        with Image.open(filename) as img:
            x_offset = (i % collage_width) * img_width
            y_offset = (i // collage_width) * img_height
            collage.paste(img, (x_offset, y_offset))
    
    # Save the collage
    collage.save(os.path.join(out_folder, output_collage_filename), "JPEG", quality=50, optimize=True)






def csv_to_sqlite(csv_filepath: str, db_name: str = "coil_elements_snr.db", table_name: str = "coil_elements_snr") -> str:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Ensure the table and unique index exist
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        Institution TEXT,
        Scanner TEXT,
        ID INTEGER,
        Date TEXT,
        Coil TEXT,
        SerialNumber TEXT,
        Element_SNR TEXT,
        UNIQUE(Institution, Scanner, ID, Date, Coil, SerialNumber)
    );
    ''')
    cursor.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_unique ON {table_name}(Institution, Scanner, ID, Date, Coil, SerialNumber);")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_filepath)

    # Define SQL statement for inserting or replacing data
    sql = f'''
    INSERT OR REPLACE INTO {table_name} 
    (Institution, Scanner, ID, Date, Coil, SerialNumber, Element_SNR) 
    VALUES (?, ?, ?, ?, ?, ?, ?);
    '''

    # Prepare data for batch operation
    data = df.values.tolist()

    # Execute batch operation
    cursor.executemany(sql, data)
    conn.commit()

    # Close the connection
    conn.close()

    return f"Database {db_name} updated."





def main(data_folder: str, out_folder: str = None, serial: str = '') -> None:
    """Main function to orchestrate the DICOM file analysis and report generation."""
    if not out_folder:
        out_folder = os.path.join(data_folder, 'SNR_test_results')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    df = analyze_dicom_files(data_folder, serial)
    reduced_df = reduce_df(df)
    print(f'\n{reduced_df}')
    csv_path = os.path.join(out_folder, 'reduced_SNR_test_results.csv')
    reduced_df.to_csv(csv_path, index=False)
    
    clean_unnecessary_images(df, out_folder)
    create_collage(out_folder)

    image_path = create_snr_chart(df)
    collage_image_path = [f for f in Path(out_folder).rglob('*') if 'collage' in f.stem][0]
    pdf_path = generate_pdf_report_reportlab(df, image_path, df['Date'].iloc[0], df['Coil'].iloc[0], df['Scanner'].iloc[0], collage_image_path, out_folder)
    print(f"\nReport saved at: {pdf_path}")
    csv_to_sqlite(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DICOM files and generate a report.")
    parser.add_argument("data_folder", type=str, help="Path to the directory containing the DICOM files.")
    parser.add_argument("--out_folder", type=str, default=None, help="Path to the directory where the results will be saved.")
    parser.add_argument("--serial", type=str, default='Unknown', help="Serial number to be added to the DataFrame.")
    parser.add_argument("--threshold_value", type=int, default=50, help="SNR threshold value used in visualizations and reporting.")
    
    args = parser.parse_args()
    main(args.data_folder, args.out_folder, args.serial)
