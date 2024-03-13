import cv2
import numpy as np
import streamlit as st

st.title("Zebrafish Fluorescence Detector")

# Choose video source and captures a frame
video_source = st.radio("Select Video Source", ("Webcam", "Upload"))
if video_source == "Webcam":
    video_capture = cv2.VideoCapture(0)
else:
    # Video upload
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read video from uploaded file
        video_capture = cv2.VideoCapture("temp_video.mp4")

if 'video_capture' in locals():
    ret, frame = video_capture.read()
    if ret:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours in the noise-reduced image
        contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour in the list of contours
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw contours and bounding rectangle on the original image
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detect fluorescence
        # Convert array to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to detect fluorescence
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Calculate the percentage of pixels that are fluorescent
        fluorescent_pixels = np.count_nonzero(binary)
        total_pixels = frame.shape[0] * frame.shape[1]
        fluorescence_level = (fluorescent_pixels / total_pixels) * 100
        st.write("Fluorescence Level: {:.2f}%".format(fluorescence_level))
        st.image(frame, channels="BGR", use_column_width=True)

        # Release the video capture
        video_capture.release()
    else:
        st.write("Failed to capture frame from video source.")
