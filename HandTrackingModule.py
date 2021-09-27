import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=True, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for handpoints in self.result.multi_hand_landmarks:
                if draw:
                  self.mpDraw.draw_landmarks(img, handpoints, self.mpHands.HAND_CONNECTIONS)
        return img

    def FindPosition(self, img, handNo=0, draw=True):

                lmlist = []
                if self.result.multi_hand_landmarks:
                    myHand = self.result.multi_hand_landmarks[handNo]
                    for id, lms in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lms.x * w), int(lms.y * h)
                        lmlist.append([id, cx, cy])
                        if draw:
                           cv2.circle(img, (cx, cy), 13, (255, 0, 255), cv2.FILLED)
                return lmlist




def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.FindHands(img)
        lmlist = detector.FindPosition(img)
        if len(lmlist) != 0:
           print(lmlist[4])

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()