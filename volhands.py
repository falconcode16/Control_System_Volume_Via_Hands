import HandTrackingModule
import cv2
import time
import numpy as np

import mediapipe as mp

import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
wCam, hCam = 740, 580
################################
class handDetector():
    def __init__(self,mode=False,maxHands = 2, detectionCon=0.5, trackCon=0.5):
      self.mode = mode
      self.maxHands = maxHands
      self.detectionCon = detectionCon
      self.trackCon = trackCon


      self.mpHands = mp.solutions.hands
      self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,
                                      self.trackCon)
      self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
   #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                   self.mpDraw.draw_landmarks(img, handLms,
                                              self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand =  self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (5, 0, 255), cv2.FILLED)

        return lmlist
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(7, wCam)
cap.set(7, hCam)
pTime = 0

detector = handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand range 50 - 300
        # Volume Range -65 - 0

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 148), (75, 400), (255, 240, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (75, 400), (255, 245, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 435), cv2.FONT_HERSHEY_DUPLEX,
                1, (0, 0, 250), 2)
    cv2.putText(img, f'Control through index finger and thumb', (2, 45), cv2.FONT_HERSHEY_DUPLEX,
                1, (0, 250, 0), 2)
    cv2.putText(img, f'Please place your palm at the centre', (2, 470), cv2.FONT_HERSHEY_DUPLEX,
                1, (250, 0, 0), 2)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (40, 120), cv2.FONT_HERSHEY_DUPLEX,
                1, (0, 0, 250), 2)

    cv2.imshow("Img", img)
    cv2.waitKey(1)