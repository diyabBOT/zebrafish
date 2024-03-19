import cv2
import streamlit as st
import numpy as np

def is_fluorescence_detected(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([30, 50, 50])
    upper_bound = np.array([90, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    num_pixels = cv2.countNonZero(mask)
    threshold = 1500

    if num_pixels > threshold:
        return True
    else:
        return False
    if True:
        total_pixels = mask.shape[0] * mask.shape[1]
    # Count the number of fluorescent pixels
    fluorescent_pixels = cv2.countNonZero(mask)
    # Calculate the percentage of fluorescent pixels
    percentage_fluorescent = (fluorescent_pixels / total_pixels) * 100
    return percentage_fluorescent
    # Calculate the percentage of pixels that are fluorescent
    st.write("Fluorescence Level: {:.2f}%".format([percentage_fluorescent]))
    st.image(frame, channels="BGR", use_column_width=True)
    return frame is not None 