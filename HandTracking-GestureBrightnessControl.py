import math
import screen_brightness_control as sbc
import mediapipe as medpie
import cv2
import numpy as np

scene = cv2.VideoCapture(0)
myhands = medpie.solutions.hands
H = myhands.Hands()
medpiedraw = medpie.solutions.drawing_utils

minBrightness =0
maxBrightness =100


while True:
    success, jolleyimg = scene.read()
    process_img = cv2.cvtColor(jolleyimg, cv2.COLOR_BGR2RGB)
    resultPoints = H.process(process_img)
    if resultPoints.multi_hand_landmarks:
        for handpoints in resultPoints.multi_hand_landmarks:
            medpiedraw.draw_landmarks(jolleyimg, handpoints, myhands.HAND_CONNECTIONS)
    handno = 0
    landmarks_list = []
    if resultPoints.multi_hand_landmarks:
        hands_detected = resultPoints.multi_hand_landmarks[handno]
        for id, lms in enumerate(hands_detected.landmark):
            h , w, c = jolleyimg.shape
            cx , cy = int(lms.x * w), int(lms.y * h)
            landmarks_list.append([id, cx, cy])

    if len(landmarks_list) != 0:
        x1, y1 = landmarks_list[4][1], landmarks_list[4][2]
        x2, y2 = landmarks_list[8][1], landmarks_list[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(jolleyimg, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(jolleyimg, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(jolleyimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(jolleyimg, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        length_btn_fingers = math.hypot(x2-x1, y2-y1)

        brightness_change = np.interp(length_btn_fingers, [50,300], [minBrightness,maxBrightness])
        sbc.set_brightness(brightness_change, None)

    colorimg = cv2.cvtColor(jolleyimg, cv2.COLOR_BGR2RGB)
    dispimg = cv2.imshow("look here!", jolleyimg)
    cv2.waitKey(1)