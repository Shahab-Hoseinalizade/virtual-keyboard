import cv2
import mediapipe as mp
import numpy as np

### Model
detection_model = mp.solutions.hands.Hands()
hand_draw = mp.solutions.drawing_utils

## Keys List
keys_list = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ' '],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']]

## Define Class For Creating Buttons
button_list = []
class Button:
    def __init__(self, pos=None, txt=None, size=[85,85]):
        self.pos = pos
        self.txt = txt
        self.size = size
        self.button_list = button_list
        
    ## Craete keyboard Table
    def create_keys(self, keys = keys_list):

        
        for i in range(len(keys)):
            for (j, key) in enumerate(keys[i]):
                self.button_list.append(Button([100* j + 50, 100 * i + 50], key))

        return button_list

    ## Draw All Buttons
    def draw_all(self, image):

        for button in self.button_list:

            x, y = button.pos
            w, h = button.size
            cv2.rectangle(image, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
            cv2.putText(image, button.txt, (x + 20, y + 65),
                        cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

        return image

    ##Draw All Buttons (Transparent)
    def draw_trans_all(self, image):

        image_new = np.zeros_like(image, np.uint8)

        for button in self.button_list:
            x, y = button.pos

            cv2.rectangle(image_new, button.pos, (x +  button.size[0], y +  button.size[1]), (255, 0, 255), cv2.FILLED)
            cv2.putText(image_new, button.txt, (x + 40, y + 60),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        out = image.copy()
        alpha = 0.5
        mask = image_new.astype(bool)
        # print(mask.shape)
        out[mask]= cv2.addWeighted(image, alpha, image_new, 1 - alpha, 0)[mask]

        return out

### Define Class For Hands Tracking And Finding Positions
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.detection_model = mp.solutions.hands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.hand_draw = mp.solutions.drawing_utils

    def handFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.detection_model.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.hand_draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])

        return lmlist
        