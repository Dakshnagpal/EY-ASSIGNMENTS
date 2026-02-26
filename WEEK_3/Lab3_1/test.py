"""
Quick test script to verify OpenCV installation and webcam access
"""
import cv2

# Display OpenCV version
print("OpenCV version:", cv2.__version__)

# Test webcam availability
cap = cv2.VideoCapture(0)
print("Webcam accessible:", cap.isOpened())
cap.release()