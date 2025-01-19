# Real-Time Instance Segmentation with YOLOv8 series and Streamlit

This project demonstrates **real-time instance segmentation** using the YOLOv8 model and a user-friendly interface built with **Streamlit**. It allows users to perform instance segmentation on live webcam feeds, uploaded images, or videos. The application provides options to select different YOLOv8 models, adjust confidence thresholds, and view object counts in real-time.

---

## Features

- **Real-Time Webcam Segmentation**: Perform instance segmentation on live webcam feeds.
- **Upload Image/Video**: Upload images or videos for segmentation.
- **Model Selection**: Choose from multiple YOLOv8 models (`yolov8n-seg`, `yolov8s-seg`, `yolov8m-seg`, `yolov8l-seg`, `yolov8x-seg`).
- **Confidence Threshold**: Adjust the confidence threshold for object detection.
- **Object Counts**: Display real-time object counts for detected classes.
- **Download Results**: Download segmented images or videos.

---

## Requirements

To run this project, you need the following dependencies:

- Python 3.8 or higher
- Streamlit
- OpenCV (`cv2`)
- Ultralytics (YOLOv8)
- Tempfile (for handling temporary files)
- Time (for FPS calculation)

You can install the required packages using the following command:

    ```bash
    pip install streamlit opencv-python ultralytics

## Installation:
Clone the repository:

    ```bash
    git clone https://github.com/alirzx/Real-time-Instance-Segmentation-Using-YOLOv8.git
    cd Real-time-Instance-Segmentation-Using-YOLOv8

## install the required dependencies:

    ```bash
    pip install -r requirements.txt

Download the YOLOv8 segmentation models (if not already included):
The application will automatically download the selected model when you run it.

## Usage
Run the Streamlit app:

    ```bash
    streamlit run app.py

Use the sidebar to:
Select a task: Real-Time Webcam or Upload Video/Image.
Choose a YOLOv8 model (default: yolov8m-seg.pt).
Adjust the confidence threshold.

## For Real-Time Webcam:

Click Start Webcam to begin segmentation.
Click Stop Webcam to stop.

## For Upload Video/Image:
Upload an image or video file.
The app will process the file and display the segmented output.
Download the segmented image or video using the provided button.


## Customization
Model Selection: You can add or remove YOLOv8 models by modifying the model_name dropdown in the app.
Confidence Threshold: Adjust the confidence threshold to control the sensitivity of object detection.
Object Counts: Modify the segment_frame function to customize how object counts are displayed.


