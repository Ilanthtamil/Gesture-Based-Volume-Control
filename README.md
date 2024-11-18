# Gesture-Based-Volume-Control

# Overview
This Python project uses OpenCV, MediaPipe, and Pycaw to control the system's audio volume based on the distance between the thumb tip and index finger tip. The gesture provides an intuitive way to adjust volume using hand movements.

# Features
Real-time hand tracking using MediaPipe's Hands module.
Calculates the Euclidean distance between thumb and index finger landmarks.
Dynamically adjusts system volume based on the distance.
Displays the distance and current volume percentage on the video feed.

# Technologies Used
Python: Core programming language.
OpenCV: For video capture and image processing.
MediaPipe: For hand gesture detection and landmark extraction.
Pycaw: For controlling system audio volume.

# How It Works
Detects hand landmarks using MediaPipe.
Calculates the distance between the thumb tip and index finger tip.
Maps the distance to the system's audio volume range.
Dynamically updates the system volume as the distance changes.
