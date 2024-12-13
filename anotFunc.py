# Import Libraries
import pandas as pd
import os
import objecttracker as ot
import Audioannotation as an
import preprocessing as pr
import xml.etree.ElementTree as ET

# Extracting video feature
def videoExtraction(FLAGS,video_path):
    dataframe = ot.main(FLAGS,video_path)
    pr.fetchVideo(dataframe, video_path)
    dataframe = pr.audioAnnotation(dataframe)
    return dataframe

# Extracting audio feature  
def audioExtraction(audio_path):
    dataframe = an.main(audio_path)
    return dataframe

# Creating CSV file
def create_unique_csv(dataframe, filename_prefix='data', folder='Result'):
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Generate a unique filename based on current timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.csv"

    # Create the full file path
    file_path = os.path.join(folder, filename)

    # Store the DataFrame in the CSV file
    csv_data = dataframe.to_csv(file_path, index=False)

    return file_path


# Creating XML file
def create_unique_xml(df, root_name='data', row_name='row', file_path='Result/Annotate_xml.xml'):
    root = ET.Element(root_name)

    for index, row in df.iterrows():
        row_element = ET.SubElement(root, row_name)

        for col_name, value in row.iteritems():
            col_element = ET.SubElement(row_element, col_name)
            col_element.text = str(value)

    tree = ET.ElementTree(root)
    tree.write(file_path, encoding='utf-8', xml_declaration=True)

    return file_path
