import cv2
import mediapipe as mp
import pyautogui

# Initializes the MediaPipe Hands solution for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Opens the webcam video capture
cap = cv2.VideoCapture(0)

# Fuctio that returns true if finger tip is higher that it's joint (pointing upwards)
def finger_is_up(hand_landmarks, finger_tip, finger_pip):
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmarks:
            # 8 = tip of index finger
            # 12 = tip fo middle finger
            # 5 = base of index finger (MCP joint)
            indexTip_y = hand_landmarks.landmark[8].y
            middleTip_y = hand_landmarks.landmark[12].y

            indexBase_y = hand_landmarks.landmark[5].y

            # Scrolls up if index fingertip is higher than it's knuckle and midlle fingers 
            # tip (it checks that the user doesn't have whole hand open)
            if indexTip_y < indexBase_y and indexTip_y < middleTip_y:
                cv2.putText(frame, "SCROLL UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                pyautogui.scroll(30)
   
            # Scrolls down if index fingertip is lower than it's knuckle and the middle fingers tip. 
            if indexTip_y > indexBase_y and indexTip_y > middleTip_y:
                cv2.putText(frame, "SCROLL DOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                pyautogui.scroll(-30)

    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()