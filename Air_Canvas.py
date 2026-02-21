import cv2
import mediapipe as mp
import numpy as np
import time

print("🌊 INDEX DRAW MODE STARTED 🌊")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480,640,3),dtype=np.uint8)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

draw_color = (255,0,255)
prev_x, prev_y = 0,0

smooth_x, smooth_y = 0,0
alpha = 0.45

# ⭐ clear timer
fist_start_time = None
CLEAR_DELAY = 1.5

# ⭐ NEW: color gesture state
color_gesture_active = False

while True:
    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        h,w,_ = frame.shape

        # fingertip position
        x = int(lm[8].x*w)
        y = int(lm[8].y*h)

        # finger states
        index_up  = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y

        fist = not index_up and not middle_up

        # smooth motion
        smooth_x = int(alpha*x + (1-alpha)*smooth_x)
        smooth_y = int(alpha*y + (1-alpha)*smooth_y)

        # ✏️ DRAW WITH INDEX FINGER ONLY
        if index_up and not middle_up:

            if prev_x==0 and prev_y==0:
                prev_x,prev_y = smooth_x,smooth_y

            for r in range(12,3,-3):
                cv2.circle(canvas,(smooth_x,smooth_y),r,draw_color,-1)

            cv2.line(canvas,(prev_x,prev_y),(smooth_x,smooth_y),draw_color,10)

            prev_x,prev_y = smooth_x,smooth_y
        else:
            prev_x,prev_y = 0,0

        # 🎨 COLOR CHANGE ONLY ONCE PER GESTURE
        if index_up and middle_up:
            if not color_gesture_active:
                draw_color = (
                    np.random.randint(0,255),
                    np.random.randint(0,255),
                    np.random.randint(0,255)
                )
                color_gesture_active = True
        else:
            color_gesture_active = False

        # 🧹 SAFE CLEAR (hold fist)
        if fist:
            if fist_start_time is None:
                fist_start_time = time.time()

            if time.time() - fist_start_time > CLEAR_DELAY:
                canvas[:] = 0
                fist_start_time = None
        else:
            fist_start_time = None

        mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

    output = cv2.add(frame,canvas)

    cv2.imshow("🌊 Index Brush Canvas 🌊",output)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()