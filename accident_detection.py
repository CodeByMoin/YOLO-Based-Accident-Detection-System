from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import numpy as np
from sort import *

vid = cv.VideoCapture("cctv_crash.mp4")

model = YOLO("best_3_m.pt")

track = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

detected_accident = {}

while True:
    r, frame = vid.read()

    if r:
        frame = cv.resize(frame, (1280, 720))
        results = model(frame, stream=True)

        detections = np.empty((0, 5))

        for b in results:
            boxes = b.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        track_results = track.update(detections)

        for res in track_results:
            x1, y1, x2, y2, ID = res
            
            if not any(np.isnan([x1, y1, x2, y2, ID])):
                x1, y1, x2, y2, ID = int(x1), int(y1), int(x2), int(y2), int(ID)
                w, h = x2 - x1, y2 - y1

                if ID not in detected_accident:
                    detected_accident[ID] = 0

                if detected_accident[ID] >= 15:
                    cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(0, 255, 0), colorC=(255, 0, 255))

                if ID in detected_accident:
                    detected_accident[ID] += 1

                if detected_accident[ID] == 15:
                    print("ACCIDENT DETECTED!!!")

        cv.imshow("img", frame)
        if cv.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cv.destroyAllWindows()