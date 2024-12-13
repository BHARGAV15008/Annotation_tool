import streamlit as st
import anotFunc as ant
from tensorflow.keras.preprocessing.sequence import pad_sequences
from absl import flags
from absl.flags import FLAGS
import soundfile as sf
import numpy as np
import os
import shutil
import sys
import tempfile
import matplotlib.pyplot as plt
import base64

#Arguments value
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-320',
                    'path to weights file')
flags.DEFINE_integer('size', 320, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('CSV',True,'store the annotating result to csv file')


# Define the Streamlit app
st.set_page_config(
    page_title="Auto Annotation Tool",
    page_icon="https://s41256.pcdn.co/wp-content/uploads/2021/03/automated-video-annotation.png",
    layout="wide"
    )
st.set_option('deprecation.showPyplotGlobalUse', False)


cascade_style = """
<style>

#root > div:nth-child(1) > div > div > div{
    margin: 5px;
    background-color: rgb(243 245 237);
    border: solid;
    border-radius: 10px;
    border-width: 2px;
    box-shadow: .5px 2px 3.5px gray;
}

.css-19or5k2.ehezqtx6.StatusWidget-enter-done {
    display: none;
}

.h1_style{
    text-align: center;
    font-family: Math;
    color: Green; 
    font-size: 52px;
    padding: 0;
    margin-bottom: 8px;
    font-weigth: bolder;
    text-shadow: 2px 2px 2px gray;
}

#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div:nth-child(4) > div > label{
    font-family: Cascadia Code;
    padding: 5px;
    display: flex;
    border: solid;
    border-width: 2px;
    border-radius: 10px;
    font-weight: 600;
    color: chocolate;
}

#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div:nth-child(4) > div > section{
    font-family: Cascadia Code;
    font-weight: 600;
    color: blue;
    border: solid;
    border-color: black;
    border-radius: 10px;
    border-width: 2px;
}

#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div:nth-child(4) > div > section > div > span > svg,
#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div:nth-child(4) > div > div > ul > li > div > div.css-10ix4kq.exg6vvm12 > svg{
    color: blue;
}

#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div:nth-child(4) > div > div > ul > li > div{
    font-family: Cascadia Code;
    padding: 5px;
    display: flex;
    border: solid;
    border-width: 2px;
    border-radius: 10px;
    font-weight: 600;
    color: chocolate;
}

.css-1syfshr.exg6vvm0,
.css-4czgeg.exg6vvm0{
    background: rgb(240 242 246);
}

hr{
    margin: 0;
    padding: 0;
    border-width: 2px;
    border-color: blue;
}


#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div:nth-child(5) > video{
    border: solid;
    border-color: gray;
    height: 500px;
    border-width: 2px;
    border-radius: 8px;
    box-shadow: 0 2px 2.5px gray;
}

.stVideo{
    border: solid;
    border-color: gray;
    height: 500px;
    border-width: 2px;
    border-radius: 8px;
    box-shadow: 0 2px 2.5px gray;
}

.css-1kyxreq.etr89bj0{
    display: contents;
    align-content: center;
    align-items: center;
}

.css-1v0mbdj.etr89bj1 img{
    border: solid;
    border-color: gray;
    height: 500px;
    border-width: 2px;
    border-radius: 8px;
    box-shadow: 0 2px 2.5px gray;
}


.stAudio{
    border: solid;
    border-color: gray;
    border-width: 2px;
    border-radius: 8px;
    box-shadow: 0 2px 2.5px gray;
}


.row-widget.stButton .css-160hik1.edgvbvh1{
    border-width: 2px;
    border-radius: 10px;
    font-family: math;
    width: 100%;
    font-weight: 600;
}

#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div.css-1ldcz5v.e1tzin5v3 > div > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div > label,
#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div.css-1ldcz5v.e1tzin5v3 > div > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div > label{
    color: blue;
    font-family: cascadia code;
    font-size: 15px;
    font-weight: 500;
}

.row-widget.stSelectbox .css-1vgnld3.effi0qh0{
    color: blue;
    font-family: cascadia code;
    font-size: 15px;
    font-weight: 500;
}

#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div.css-1ldcz5v.e1tzin5v3 > div > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div > div > div,
#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div.css-1ldcz5v.e1tzin5v3 > div > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div > div > div{
    border: none;
    color: orange;
    font-family: math;
    font-weight: 400;
    box-shadow: 1px 2px 2.5px black
}

.st-ae.st-af.st-ag.st-ah.st-ai.st-aj.st-bw.st-bo.st-b7{
    border: none;
    color: orange;
    font-family: math;
    font-weight: 500;
    background: white;
    border-radius: 10px;
    box-shadow: 1px 2px 2.5px black;
}

#bui-17 > div > div > ul,
#bui-18 > div > div > ul{
    color: orange;
    font-family: math;
    font-weight: 400;
    box-shadow: 1px 2px 2.5px black;
    border-radius: 10px;
}


.row-widget.stButton .css-1ubkpyc.edgvbvh1,
.row-widget.stButton .css-qbe2hs.edgvbvh1{
    padding: 6px;
    width: 100%;
    font-family: math;
    font-weight: 600;
    font-size: 18px;
    letter-spacing: 1.5px;
    color: blue;
    background-color: white;
    box-shadow: 1px 2px 3px black;
    border-width: 2px;
    border-radius: 10px;
}

.row-widget.stButton .css-1ubkpyc.edgvbvh1:hover,
.row-widget.stButton .css-qbe2hs.edgvbvh1:hover{
    color: rgb(246, 51, 102);
    border-color: rgb(246, 51, 102);
    box-shadow: 1px 2px 3px rgb(246, 51, 102);
    text-shadow: 1px 1px black;

}

.element-container.css-1e5imcs.e1tzin5v1,
.stMarkdown{
    margin-top: 10px;
    margin-bottom: 5px;
}

.row-widget.stDownloadButton .css-1ubkpyc.edgvbvh1,
.row-widget.stDownloadButton .css-qbe2hs.edgvbvh1,
.downloadBtn{
    text-decoration: none;
    text-align: center;
    padding:10px;
    width: 30%;
    font-family: math;
    font-weight: 600;
    color: blue;
    background-color: white;
    border: none;
    box-shadow: 1px 2px 3px blue;
    border-radius: 10px;
    letter-spacing: 1px;
}

.row-widget.stDownloadButton .css-1ubkpyc.edgvbvh1,
.row-widget.stDownloadButton .css-qbe2hs.edgvbvh1,
.downloadBtn *:not(:last-child) {
  margin-right: 0;
  margin-left: 0;
}


.arr_dl{
    width:100%;
    display: flex;
    justify-content: space-evenly;
    flex-wrap: nowrap;
    flex-direction: row;
    align-items: center;
}

.row-widget.stDownloadButton .css-1ubkpyc.edgvbvh1:hover,
.row-widget.stDownloadButton .css-qbe2hs.edgvbvh1:hover,
.downloadBtn:hover{
    text-decoration: none;
    border: solid 2px;
    color: green;
    border-color: green;
    box-shadow: 1px 2px 3px green;
    text-shadow: 1px 1px rgb(28 182 247)
}


#root > div:nth-child(1) > div > div > div > div > section > footer{
    display: none;
}

</style>
"""
FLAGS = flags.FLAGS
FLAGS(sys.argv)

# Display a title with custom CSS style
st.markdown(cascade_style, unsafe_allow_html=True)
st.write("<h1 class='h1_style'>Auto Annotation Tool</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


def convert_to_wav(input_file, output_file):
    audio, sr = sf.read(input_file)
    sf.write(output_file, audio, sr)


def plot_parameter(df, column_name, title):
    plt.figure(figsize=(10, 5))
    parameter_arr = np.array(df[column_name].tolist(), dtype=object)

    # Pad the inner sequences to match the maximum length
    parameter_arr = pad_sequences(parameter_arr, dtype=complex, padding='post')

    # Convert complex values to absolute values
    parameter_arr = np.abs(parameter_arr)

    plt.imshow(parameter_arr.T, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.title(title)
    plt.xlabel('Segment')
    plt.ylabel('Feature Dimension')
    return plt


def videoAnnotation():
    # Create directory for annotate files
    if not os.path.exists("Result"):
        os.makedirs("Result")

    temp_file_path = ""
    clicked = False
    uploaded_file = st.file_uploader("Annotate Your Video...", type=['mp4', 'avi', 'mpg', 'mov', 'mkv', 'wmv', 'mp3', 'wav', 'aac', 'wma'])
    
    plHolder = st.empty()

    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    col1, col2 = st.columns(2)

    with plHolder:
        # csv = ""
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1]

            if file_extension.lower() in ['.mp3', '.wav', '.aac', '.wma']:
                clicked = col2.button("Audio Extraction")
                
                st.audio(uploaded_file, format='audio/*', start_time=0)
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_filename = temp_file.name
                    temp_file.write(uploaded_file.read())

                output_file = "output_audio.wav"  # Replace with your desired output file path
                convert_to_wav(temp_filename, output_file)

                if clicked:
                    json_file = "Result/Annotate_json.json"

                    dataframe = ant.audioExtraction(output_file)
                    csv_file = ant.create_unique_csv(dataframe)
                    df_json = dataframe.copy()
                    # Convert DataFrame to JSON
                    df_json['MFCC Parameters'] = df_json['MFCC Parameters'].apply(lambda x: str(x))
                    df_json['STFT_Parameters'] = df_json['STFT_Parameters'].apply(lambda x: str(x))
                    df_json['Bark Parameters'] = df_json['Bark Parameters'].apply(lambda x: str(x))
                    df_json['CWT Parameters'] = df_json['CWT Parameters'].apply(lambda x: str(x))

                    json_data = df_json.to_json(orient='records')

                    # Write JSON data to a file 
                    with open(json_file, 'w') as file:
                        file.write(json_data)
                        
                    xml_file = ant.create_unique_xml(dataframe)

                    plts = plot_parameter(dataframe, "MFCC Parameters", "MFCC Parameters")
                    col3.pyplot(plts)
                    plts = plot_parameter(dataframe, "STFT_Parameters", "STFT_Parameters")
                    col4.pyplot(plts)
                    plts = plot_parameter(dataframe, "CWT Parameters", "CWT Parameters")
                    col5.pyplot(plts)
                    plts = plot_parameter(dataframe, "Bark Parameters", "Bark Parameters")
                    col6.pyplot(plts)

                    with open(csv_file, "rb") as file:
                        csv_contents = file.read()

                    with open(json_file, "rb") as file:
                        json_contents = file.read()

                    with open(xml_file, "rb") as file:
                        xml_contents = file.read()

                    col2.markdown(
                        f'<div class="arr_dl">'
                        f'<a href="data:application/octet-stream;base64,{base64.b64encode(csv_contents).decode()}" '
                        'download="Annotate_csv.csv" class="downloadBtn">CSV File</a>'
                        f'<a href="data:application/octet-stream;base64,{base64.b64encode(json_contents).decode()}" '
                        'download="Annotate_json.json" class="downloadBtn">JSON File</a>'
                        f'<a href="data:application/octet-stream;base64,{base64.b64encode(xml_contents).decode()}" '
                        'download="annotate_xml.xml" class="downloadBtn">XML File</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    

            elif file_extension.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']:
                temp_dir = tempfile.TemporaryDirectory()
                temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                clicked = col1.button("Video Extraction")
                if not clicked:
                    if uploaded_file is not None:
                        st.video(uploaded_file, start_time=0)
                else:
                    json_file = "Result/Annotate_json.json"
                    dataframe = ant.videoExtraction(FLAGS,temp_file_path)
                    csv_file = ant.create_unique_csv(dataframe)
                    xml_file = ant.create_unique_xml(dataframe)

                    dataframe['MFCC FEATURE'] = dataframe['MFCC FEATURE'].apply(lambda x: str(x))
                    dataframe['STFT FEATURE'] = dataframe['STFT FEATURE'].apply(lambda x: str(x))
                    dataframe['BARK FEATURE'] = dataframe['BARK FEATURE'].apply(lambda x: str(x))
                    dataframe['CWT FEATURE'] = dataframe['CWT FEATURE'].apply(lambda x: str(x))

                    json_data = dataframe.to_json(orient='records')

                    # Write JSON data to a file 
                    with open(json_file, 'w') as file:
                        file.write(json_data)

                    with open(csv_file, "rb") as file:
                        csv_contents = file.read()

                    with open(json_file, "rb") as file:
                        json_contents = file.read()

                    with open(xml_file, "rb") as file:
                        xml_contents = file.read()

                    col1.markdown(
                        f'<div class="arr_dl">'
                        f'<a href="data:application/octet-stream;base64,{base64.b64encode(csv_contents).decode()}" '
                        'download="Annotate_csv.csv" class="downloadBtn">CSV File</a>'
                        f'<a href="data:application/octet-stream;base64,{base64.b64encode(json_contents).decode()}" '
                        'download="Annotate_json.json" class="downloadBtn">JSON File</a>'
                        f'<a href="data:application/octet-stream;base64,{base64.b64encode(xml_contents).decode()}" '
                        'download="annotate_xml.xml" class="downloadBtn">XML File</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
               
            else:
                st.write("Uploaded file is neither an audio nor a video file.")
            
            # clean the directory
            shutil.rmtree("Result")


if __name__ == '__main__':
    videoAnnotation()
