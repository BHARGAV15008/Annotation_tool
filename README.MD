# Audio-Video Annotation

**The Audio-Video Annotation and Feature Extraction Project is an innovative and advanced tool designed to analyze video content and extract valuable features for both visual and auditory elements. The project utilizes state-of-the-art algorithms to identify individual objects in videos, pinpoint their entry and exit frames, and provide accurate timestamps. Furthermore, the application goes beyond visual analysis by extracting various audio features, including Mel-Frequency Cepstral Coefficients (MFCC), Continuous Wavelet Transform (CWT), Short-Time Fourier Transform (STFT), and Bark features, for each object detected in the video.**

**Objectives**

***Video Analysis: The project's primary objective is to conduct comprehensive video analysis. By employing sophisticated computer vision techniques, the system identifies and tracks individual objects throughout the video.***

***Entry and Exit Frames: The application determines the precise frames at which each object enters and exits the video, enabling detailed behavioral analysis and understanding of object movements.***

***Timestamp Generation: In addition to identifying entry and exit frames, the system generates accurate timestamps for each object's appearance and disappearance in the video.***

***Audio Feature Extraction: The project incorporates advanced audio processing methods to extract a range of features from the video's audio track. This includes MFCC, CWT, STFT, and Bark features, providing valuable insights into the audio characteristics of each identified object.***

***Object-Specific Analysis: The extracted audio features are linked to the identified objects, allowing users to understand the audio attributes associated with individual objects in the video.***


## Getting Started

**To run this project, create a Python environment using a virtual environment. Follow the steps below:**

1. Install `virtualenv` using the following Python `pip` command:

```bash
    pip install virtualenv
```


2. Create a virtual environment by executing the command below (replace `<version>` with your Python version and `<virtual-environment-name>` with your desired name):

```bash
    python3.8 -m venv <virtual-environment-name>
```


3. Activate the virtual environment:

```bash
    source env/bin/activate
``` 
    - in unix {mac/linux}

```bash
    env\Scripts\activate.bat
``` 
    - in cmd

```bash
    env\Scripts\Activate.ps1
``` 
    - in powershell


4. Confirm that the environment is active by checking installed packages:

```bash
    pip list
```
*The active environment should display only two packages while showing many packages elsewhere.*



5. Install all the required libraries for this project:

```bash
    pip install -r requirements.txt
```


## Running the Application

*After installing the required libraries, run the `webApp.py` Python file to launch the application:*

```bash
    python webApp.py
    streamlit run webApp.py
```

## Deactivating the virtual environment

*Once you finish using the application, deactivate the active environment:*
    
```bash
    deactivate
```

***Thank you for exploring our project and using our application! If you encounter any issues or have any feedback, please feel free to reach out. Happy annotating!***