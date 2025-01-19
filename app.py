import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import os

# Streamlit app title
st.title("Real-Time Instance Segmentation with YOLOv8")

# Sidebar for options
st.sidebar.header("Options")
task = st.sidebar.radio("Choose Task", ["Real-Time Webcam", "Upload Video/Image"])

# Model selection
model_name = st.sidebar.selectbox(
    "Select YOLOv8 Model",
    ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
    index=2,  # Default to YOLOv8m-seg
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

# Load the selected YOLOv8 model
model = YOLO(model_name)

# Function to perform instance segmentation on a frame and calculate FPS/object counts
def segment_frame(frame, prev_time):
    # Perform instance segmentation with confidence threshold
    results = model(frame, conf=confidence_threshold)
    annotated_frame = results[0].plot()

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Get detected objects and count them
    detections = results[0].boxes.data
    class_counts = {}
    for detection in detections:
        class_id = int(detection[-1])
        class_name = model.names[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Draw FPS and object counts on the frame
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for i, (class_name, count) in enumerate(class_counts.items()):
        cv2.putText(annotated_frame, f"{class_name}: {count}", (10, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated_frame, prev_time, class_counts

# Real-Time Webcam Task
if task == "Real-Time Webcam":
    st.header("Real-Time Webcam Segmentation")
    start_button = st.button("Start Webcam")

    if start_button:
        st.write("Webcam started. Press 'Stop Webcam' to stop.")
        stop_button = st.button("Stop Webcam")

        # Open the webcam
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        prev_time = 0

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            # Perform instance segmentation and get FPS/object counts
            annotated_frame, prev_time, class_counts = segment_frame(frame, prev_time)

            # Display the annotated frame
            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

            # Display object counts in the sidebar
            st.sidebar.header("Object Counts (Real-Time)")
            for class_name, count in class_counts.items():
                st.sidebar.write(f"{class_name}: {count}")

            # Check if the stop button is pressed
            if stop_button:
                break

        # Release the webcam
        cap.release()
        st.write("Webcam stopped.")

# Upload Video/Image Task
elif task == "Upload Video/Image":
    st.header("Upload Video or Image for Segmentation")
    uploaded_file = st.file_uploader("Upload a video or image", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name

        # Check if the file is an image or video
        if uploaded_file.type.startswith("image"):
            # Process image
            st.write("Processing image...")
            frame = cv2.imread(file_path)
            prev_time = time.time()
            annotated_frame, _, class_counts = segment_frame(frame, prev_time)

            # Display the segmented image
            st.image(annotated_frame, channels="BGR", caption="Segmented Image", use_column_width=True)

            # Display object counts in the sidebar
            st.sidebar.header("Object Counts (Image)")
            for class_name, count in class_counts.items():
                st.sidebar.write(f"{class_name}: {count}")

            # Add a download button for the segmented image
            if st.button("Download Segmented Image"):
                output_path = "segmented_image.jpg"
                cv2.imwrite(output_path, annotated_frame)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Image",
                        data=file,
                        file_name="segmented_image.jpg",
                        mime="image/jpeg",
                    )
                os.remove(output_path)

        elif uploaded_file.type.startswith("video"):
            # Process video
            st.write("Processing video...")
            cap = cv2.VideoCapture(file_path)
            frame_placeholder = st.empty()
            prev_time = time.time()

            # Create a temporary file for the segmented video
            output_video_path = "segmented_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform instance segmentation and get FPS/object counts
                annotated_frame, prev_time, class_counts = segment_frame(frame, prev_time)

                # Write the annotated frame to the output video
                out.write(annotated_frame)

                # Display the annotated frame
                frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

                # Display object counts in the sidebar
                st.sidebar.header("Object Counts (Video)")
                for class_name, count in class_counts.items():
                    st.sidebar.write(f"{class_name}: {count}")

                # Add a small delay to simulate real-time playback
                time.sleep(0.03)

            # Release the video capture and writer
            cap.release()
            out.release()
            st.write("Video processing complete.")

            # Add a download button for the segmented video
            if st.button("Download Segmented Video"):
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file,
                        file_name="segmented_video.mp4",
                        mime="video/mp4",
                    )
                os.remove(output_video_path)