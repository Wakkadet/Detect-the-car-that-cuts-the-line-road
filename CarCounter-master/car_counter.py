from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

#code Trained 
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt","MobileNetSSD_deploy.caffemodel")

#open video
print("[INFO] opening video file...")
vs = cv2.VideoCapture("16.mp4")

#star frame to none
#they will be change when analyzing the1st frame for make faster
W = None
H = None

#start algorithm
###maxDisappeared = จำนวนเฟรมที่วัตถุสามารถหายไปจากวิดีโอจากนั้นอีกครั้ง
###maxDistance = ระยะห่างสูงสุดระหว่างจุดศูนย์กลางของวงกลมที่ถูกบันทึกไว้ในกล่องรถยนต์
###หากระยะทางน้อยกว่าที่ระบุ ID จะถูกกำหนดใหม่
ct = CentroidTracker(maxDisappeared=10, maxDistance=40)
trackers = []
trackableObjects = {}
startObject = []
checkObject = []
# จำนวนเฟรมทั้งหมดในวิดีโอ
totalFrames = 0
totalcar = 0
# สถานะ:การติดตาม
status = None

# สร้างกราฟที่แสดงจำนวนรถยนต์ในวิดีโอที่เพิ่มขึ้น
frames = []

# หมายเลขเฟรมวิดีโอ
frame_number = 0

# แต่ละเฟรมของวิดีโอ
while True:
        frame_number += 1
        frames.append(frame_number)
        frame = vs.read()
        frame = frame[1]

        # หากเฟรมเป็นค่าว่างแสดงว่าถึงจุดสิ้นสุดของวิดีโอแล้ว
        if frame is None:
                print("=============================================")
                print("The end of the video reached")
                print("Total number of cars on the video is ", len(startObject))
                print("=============================================")
                break

        # ปรับขนาดเฟรมเพื่อเพิ่มความเร็วในการทำงาน
        frame = imutils.resize(frame, width=800)

        # เปลี่ยนสีเป็น RGB แทน BGR เพื่อให้ไลบรารี dlib 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ขนาดเฟรม
        if W is None or H is None:
                (H, W) = frame.shape[:2]


        rects = []

        if totalFrames % 3 == 0:
                trackers = []

                status = "Detecting..."

                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)

                detections = net.forward()

                for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]

                        if confidence > 0.4:
                                idx = int(detections[0, 0, i, 1])

                                if CLASSES[idx] != "car" and CLASSES[idx] != "motorbike":
                                        continue
                                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                                (startX, startY, endX, endY) = box.astype("int")
                                
                                tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(startX, startY, endX, endY)
                                tracker.start_track(rgb, rect)

                                trackers.append(tracker)

        else:
                for tracker in trackers:

                        status = "Tracking..."

                        tracker.update(rgb)

                        pos = tracker.get_position()
                        
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())

                        rects.append((startX, startY, endX, endY))

        objects = ct.update(rects)
        
        for (objectID, centroid) in objects.items():
                
                to = trackableObjects.get(objectID, None)
                
                if to is None:
                        to = TrackableObject(objectID, centroid)

                trackableObjects[objectID] = to
                   
                #วาดจุดตามรถที่ track
                text = "ID {}".format(objectID+1)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                #ตรวจสอบ ID ของรถ
                if objectID >= len(startObject):
                         startObject.append((centroid[0],centroid[1]))
                         checkObject.append(True)
                else:
                         if(startObject[objectID][0] < 370 < centroid[0] and checkObject[objectID] == True):
                                 totalcar+=1
                                 checkObject[objectID] = False
                                 cv2.imwrite(os.getcwd()  +'\car'+ str(totalcar) + '.png', frame)
                                 print(os.getcwd())
                
        info = [("TotalAllcar",len(startObject)),("Totalcar", totalcar),("Status", status)]
        
        #เขียน text
        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        #เพิ่มเส้นสีแดง
        cv2.line(frame, (370, 0), (370, 550), (0, 0, 255), 3) 

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
                print("[INFO] process finished by user")
                print("Total number of cars on the video is ",len(startObject))
                break

        # เพิ่ม frame
        totalFrames += 1

#ปิดโปรแกรม
cv2.destroyAllWindows()
