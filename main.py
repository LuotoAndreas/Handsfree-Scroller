import cv2
import mediapipe as mp
import pyautogui
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

mouth_open_threshold = 0.03
eye_closed_threshold = 0.009

closed_frames_threshold = 3
closed_frames_count = 0

def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]

            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]

            mouth_open = lower_lip.y - upper_lip.y
            left_eye_open = left_eye_bottom.y - left_eye_top.y
            right_eye_open = right_eye_bottom.y - right_eye_top.y

            if mouth_open > mouth_open_threshold:
                closed_frames_count = 0
                cv2.putText(frame, "DOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.scroll(-20)
            else:
                left_eye_closed = left_eye_open < eye_closed_threshold
                right_eye_closed = right_eye_open < eye_closed_threshold

                # Rullaa ylös vain jos oikea silmä kiinni ja vasen auki
                if right_eye_closed and not left_eye_closed:
                    closed_frames_count += 1
                else:
                    closed_frames_count = 0

                if closed_frames_count >= closed_frames_threshold:
                    cv2.putText(frame, "UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    pyautogui.scroll(30)

    cv2.imshow('Face Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()