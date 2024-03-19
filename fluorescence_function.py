import cv2
import streamlit as st

def is_fluorescence_detected(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to detect fluorescence
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Calculate the percentage of pixels that are fluorescent
    fluorescent_pixels = np.count_nonzero(binary)
    total_pixels = frame.shape[0] * frame.shape[1]
    fluorescence_level = (fluorescent_pixels / total_pixels) * 100
    st.write("Fluorescence Level: {:.2f}%".format(fluorescence_level))
    st.image(frame, channels="BGR", use_column_width=True)
    return frame is not None 