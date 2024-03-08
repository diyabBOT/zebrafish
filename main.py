# to run the code, enter 'streamlit run main.py' in the VSCode terminal 

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av 


def detect_zebrafish(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply EDGE DETECTION to detect zebrafish
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours of objects in the frame
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw a rectangle around the detected zebrafish
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Capture an image once zebrafish is detected
        if w > 50 and h > 50:  # Adjust these thresholds based on your zebrafish size
            return True, frame[y:y+h, x:x+w]
    
    return False, frame

class VideoProcessor:
    def recv(self, vid_frame):
        frm = vid_frame.to_ndarray(format='bgr24')
        
        detect_zebrafish(frm)
        #detect_zebrafish returns False, frame

        return av.VideoFormat.from_ndarray(frm, format='bgr24')

# every frame will be processed by VideoProcessor 
webrtc_streamer(key='key', video_processor_factory=VideoProcessor)


def detect_fluorescence(frame):
    # Convert array to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to detect fluorescence
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Calculate the percentage of pixels that are fluorescent
    fluorescent_pixels = np.count_nonzero(binary)
    total_pixels = frame.shape[0] * frame.shape[1]
    fluorescence_level = (fluorescent_pixels / total_pixels) * 100
    
    return fluorescence_level

def main():
    st.title("Zebrafish Fluorescence Detector")
    #write code so user can choose to provide video through webcam or users can upload a video

    # Choose video source
    video_source = st.radio("Select Video Source", ("Webcam", "Upload"))
    if video_source == "Webcam":
        # Video capture from webcam
        video_capture = cv2.VideoCapture(0)
    else:
        # Video upload
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if uploaded_file is not None:
            # Read video from uploaded file
            video_capture = cv2.VideoCapture(uploaded_file)

    # Main app loop
    while True:
        ret, frame = video_capture.read()  # ret is boolean
        # Detect zebrafish
        zebrafish_detected, zebrafish_frame = detect_zebrafish(frame)
        if zebrafish_detected:
            # Perform fluorescence detection on zebrafish frame
            fluorescence_level = detect_fluorescence(zebrafish_frame)
            # Display fluorescence level
            st.write("Fluorescence Level: {:.2f}%".format(fluorescence_level))
        # Display frame with or without zebrafish detected
        if zebrafish_detected:
            frame = zebrafish_frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", use_column_width=True)
        # Check for app exit
        if st.button("Exit", key="exit_button"):
            break
    # Release video capture
    video_capture.release()
    # Video capture
    video_capture = cv2.VideoCapture(0)

if __name__ == "__main__":
    main()




