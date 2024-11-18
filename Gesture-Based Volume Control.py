import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Pycaw for audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get volume range
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]  # Minimum volume (-65.25)
max_volume = volume_range[1]  # Maximum volume (0.0)

# Start webcam capture
cap = cv2.VideoCapture(0)

# Hand detection configuration
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Draw the hand annotations on the frame
        frame.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get thumb tip and index finger tip landmarks
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                index_tip = hand_landmarks.landmark[8]  # Index finger tip

                # Calculate Euclidean distance between thumb and index finger
                distance = np.sqrt(
                    (index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2
                )

                # Normalize distance to volume range (distance is usually between 0 and ~0.2)
                scaled_volume = np.interp(distance, [0, 0.2], [min_volume, max_volume])

                # Set system volume
                volume.SetMasterVolumeLevel(scaled_volume, None)

                # Display the distance and corresponding volume on the frame
                cv2.putText(frame, f'Distance: {distance:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f'Volume: {int(np.interp(scaled_volume, [min_volume, max_volume], [0, 100]))}%', 
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Hand Gesture Volume Control', frame)

        # Check if the window is closed
        if cv2.getWindowProperty('Hand Gesture Volume Control', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
