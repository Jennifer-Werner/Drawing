import cv2
import mediapipe as mp
import numpy as np
import time


class handDetector():

    def __init__(self, mode=False, maxHands=2, modelC=1, minDet=0.5, minTrack=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.minDetCon = minDet
        self.minTrack = minTrack
        self.modelC = modelC

        self.hands = mp.solutions.hands
        self.handsDetector = self.hands.Hands(self.mode, self.maxHands, modelC, self.minDetCon, self.minTrack)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        self.results = self.handsDetector.process(img)

        if self.results.multi_hand_landmarks:
            for landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, landmark, self.hands.HAND_CONNECTIONS)

        return img

    def trackHands(self, img, num=0, draw=True):

        self.landmark_list = []

        if self.results.multi_hand_landmarks:

                for idx, ldmk in enumerate(self.results.multi_hand_landmarks[num].landmark):
                    height, width, channel = img.shape
                    channel_x, channel_y = int(ldmk.x * width), int(ldmk.y * height)
                    self.landmark_list.append([idx, channel_x, channel_y])
                    print([idx, channel_x, channel_y])

                    if draw:
                        if idx == 8:
                            cv2.circle(img, (channel_x, channel_y), 15, (255, 0, 0), -1)

        return self.landmark_list

    def HandsUp(self):

        fingers = []

        if len(self.landmark_list) != 0:
            if self.landmark_list[4][1] < self.landmark_list[3][1]:
                fingers.append(True)
            else:
                fingers.append(False)
            for tip in range(8, 21, 4):
                if self.landmark_list[tip][2] < self.landmark_list[tip-1][2]:
                    fingers.append(True)
                else:
                    fingers.append(False)

        return fingers




def main():

    prevT = 0
    conT = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:

        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
        flipped = np.fliplr(frame)
        flipped_RGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        flipped_RGB = detector.findHands(flipped_RGB)
        landmark_list = detector.trackHands(flipped_RGB)
        if len(landmark_list) != 0:
            print(landmark_list[8])

        conT = time.time()
        fps = 1 / (conT - prevT)
        prevT = conT

        cv2.putText(flipped_RGB, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        res_image = cv2.cvtColor(flipped_RGB, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', res_image)


if __name__ == '__main__':
    main()
