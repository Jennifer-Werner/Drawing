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

while True:
    ret, frame = cap.read()
    frame[:62, :640] = img_list[0]
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

    cv2.putText(flipped_RGB, str(int(fps)), (580, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    res_image = cv2.cvtColor(flipped_RGB, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', res_image)


