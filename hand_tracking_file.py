import cv2
import mediapipe as mp
import numpy as np
import time


handsDetector, hands = mp.solutions.hands.Hands(), mp. solutions.hands
cap = cv2.VideoCapture(0)

prevT = 0
conT = 0

while cap.isOpened():

    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flipped_RGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flipped_RGB)
    mpDraw = mp.solutions.drawing_utils

    if results.multi_hand_landmarks:

        for landmark in results.multi_hand_landmarks:

            for idx, ldmk in enumerate(landmark.landmark):
                height, width, channel = flipped_RGB.shape
                channel_x, channel_y = int(ldmk.x * width), int(ldmk.y * height)

                if idx == 8:
                    cv2.circle(flipped_RGB, (channel_x, channel_y), 15, (255, 0, 0), -1)
                    print(channel_x, channel_y)

            mpDraw.draw_landmarks(flipped_RGB, landmark, hands.HAND_CONNECTIONS)

    conT = time.time()
    fps = 1 / (conT - prevT)
    prevT = conT

    cv2.putText(flipped_RGB, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    res_image = cv2.cvtColor(flipped_RGB, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', res_image)

handsDetector.close()
