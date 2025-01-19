# Real-Time Instance Segmentation with YOLOv8 and Streamlit

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

Installation:
Clone the repository:
git clone https://github.com/alirzx/Real-time-Instance-Segmentation-Using-YOLOv8.git
cd Real-time-Instance-Segmentation-Using-YOLOv8
