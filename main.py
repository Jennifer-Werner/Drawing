from hand_tracking_module import *
import os
import time

#FPS
prevT = 0
conT = 0

cap = cv2.VideoCapture(0)
detector = handDetector()

#Путь к изображениям "Меню"
file = os.path.join(os.path.dirname(__file__), 'Menu')
img_list = []
for num in os.listdir(file):
    img = cv2.imread(os.path.join(file, num))
    img = cv2.resize(img, (640, 62))
    img_list.append(img)
menu = img_list[0]

while True:

    #Обрабатываем изображение
    ret, frame = cap.read()
    frame[:62, :640] = menu
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flipped_RGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    #Отслеживаем положение рук и отрисовываем их
    flipped_RGB = detector.findHands(flipped_RGB)
    landmark_list = detector.trackHands(flipped_RGB)

    #Проверяем какие пальцы подняты
    tips = detector.HandsUp()

    if len(tips) != 0:

        #Selection Mode
        if tips[1] and tips[2] and not tips[0]:
            cv2.putText(flipped_RGB, 'Selection Mode', (500, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
            img, menu = detector.Selection_Mode(flipped_RGB, img_list)

        #Drawing Mode
        elif tips[1] and not tips[2] and not tips[0]:

            #Erasing Mode
            if detector.color == [220, 220, 220]:
                cv2.putText(flipped_RGB, 'Erasing Mode', (500, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)

            #Drawing Mode
            else:
                cv2.putText(flipped_RGB, 'Drawing Mode', (500, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
            img = detector.Drawing_Mode(flipped_RGB)

        #Size Mode
        elif tips[1] and tips[0] and not tips[2]:
            cv2.putText(flipped_RGB, 'Size Mode', (500, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
            img = detector.Size_Mode(flipped_RGB)

    #FPS
    conT = time.time()
    fps = 1 / (conT - prevT)
    prevT = conT

    cv2.putText(flipped_RGB, str(int(fps)), (580, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    #Обрабатываем и показываем изображение

    gray = cv2.cvtColor(detector.drawing, cv2.COLOR_BGR2GRAY)
    ret, inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    flipped_RGB = cv2.bitwise_and(flipped_RGB, inv)
    flipped_RGB = cv2.bitwise_or(flipped_RGB, detector.drawing)

    res_image = cv2.cvtColor(flipped_RGB, cv2.COLOR_RGB2BGR)
    drawing = cv2.cvtColor(detector.drawing, cv2.COLOR_RGB2BGR)
    cv2.imshow('Drawing', drawing)
    cv2.imshow('Image', res_image)
