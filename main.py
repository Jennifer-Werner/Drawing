from hand_tracking_module import *
import os

prevT = 0
conT = 0
cap = cv2.VideoCapture(0)

detector = handDetector()
file = os.path.join(os.path.dirname(__file__), 'Menu')
img_list = []
for num in os.listdir(file):
    img = cv2.imread(os.path.join(file, num))
    img = cv2.resize(img, (640, 62))
    img_list.append(img)
menu = img_list[0]

while True:
    ret, frame = cap.read()
    frame[:62, :640] = menu
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flipped_RGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    flipped_RGB = detector.findHands(flipped_RGB)
    landmark_list = detector.trackHands(flipped_RGB)

    if len(landmark_list) != 0:
        print(landmark_list[8])

    tips = detector.HandsUp()

    if len(tips) != 0:
        if tips[1] and tips[2]:
            cv2.putText(flipped_RGB, 'Selection Mode', (500, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
            img, menu = detector.Selection_Mode(flipped_RGB, img_list)

        if tips[1] and not tips[2]:
            if detector.color == [220, 220, 220]:
                cv2.putText(flipped_RGB, 'Erasing Mode', (500, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
            else:
                cv2.putText(flipped_RGB, 'Drawing Mode', (500, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
            img = detector.Drawing_Mode(flipped_RGB)

    conT = time.time()
    fps = 1 / (conT - prevT)
    prevT = conT

    cv2.putText(flipped_RGB, str(int(fps)), (580, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    res_image = cv2.cvtColor(flipped_RGB, cv2.COLOR_RGB2BGR)
    drawing = cv2.cvtColor(detector.drawing, cv2.COLOR_RGB2BGR)
    cv2.imshow('Drawing', drawing)
    cv2.imshow('Image', res_image)
