import cv2
import mediapipe as mp
import numpy as np
from time import sleep
import math
from codes import Button, handTracker

### Frame Size
cap = cv2.VideoCapture(0)               #Laptop Camera
cap.set(3, 1280)                        #Width Size                            
cap.set(4, 720)							#Height Size

final_txt = ''

handTracker = handTracker()
Button = Button()
clicked = ['1', '2']

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = handTracker.handFinder(frame)
    lmList = handTracker.positionFinder(frame)

    buttons = Button.create_keys()
    frame = Button.draw_trans_all(frame)

    if lmList:  					#Mediapipe
        for button in buttons:
            x, y = button.pos
            w, h = button.size
            

            if (x < lmList[8][1] < x + w) and (y < lmList[8][2] < y + h):
                cv2.rectangle(frame, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, button.txt, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                
                length = math.hypot((lmList[8][1] - lmList[12][1]), (lmList[8][2] - lmList[12][2]))
                print(length)

                ##Click
                if length < 30:

                    cv2.rectangle(frame, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, button.txt, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    clicked.append(button.txt)

                    if clicked[-2] != button.txt:
                
                        final_txt += button.txt
                        
                    print(final_txt)
                    
    cv2.rectangle(frame, (50, 350), (700, 450), (170, 0, 255), cv2.FILLED)
    cv2.putText(frame, final_txt, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(30)               # waitKey --> Secound Fpr Each Frame and Play Speed
    if key == 27:                       # ord('esc') = 27
        break
