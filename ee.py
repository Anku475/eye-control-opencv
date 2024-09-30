import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    landmarks_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmarks_points:
        landmarks = landmarks_points[0].landmark

        # Draw eye landmarks and control mouse
        for id, landmark in enumerate(landmarks[474:478]):  # Eye landmarks
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:  # Use the second eye landmark for cursor position
                screen_x = screen_w * x / frame_w
                screen_y = screen_h * y / frame_h
                pyautogui.moveTo(screen_x, screen_y)

        # Detect blink using landmarks
        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (left_eye[0].y - left_eye[1].y) < 0.004:  # Simple blink detection
            pyautogui.click()  # Simulate mouse click

    cv2.imshow('Eye Control Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
