import argparse
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from datetime import datetime
import os

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="path to the YOLO model file")
parser.add_argument("--classes", type=str, required=True, help="path to the classes file")
parser.add_argument("--video", type=str, required=True, help="path to the input video file")
args = parser.parse_args()

# model initialize
model = YOLO(args.model)

# current time for attendance file
now = datetime.now()
current_time = now.strftime("%I:%M:%S %p")
current_day= datetime.today().strftime('%Y-%m-%d')

# reading the class names from custom created classes
with open(args.classes, "r") as f:
    class_list = [c.strip() for c in f.read().split("\n") if c.strip()]

# to save the cropped images of attendance
if not os.path.exists('cropped_images'):
    os.makedirs('cropped_images')

# capturing video
cap = cv2.VideoCapture(args.video)

# seting points for zone
pnts = [(900, 500), (1200, 500), (1200,1000), (900, 1000)]
polygon = Polygon(pnts)
rectangle_st = (900,500)
rectangle_ed = (1200,1000)

# flag to check the class is crop only 1 time
marked = set()

# starting the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # getting the bounding box from tensor
    results = model.predict(frame)
    boxes = results[0].boxes    
    a = boxes.data.cpu()

    # appending it in this dataframe for futhur use and representation   
    df = pd.DataFrame(a).astype("int")

    # loop will be getting the information of bounding box using the pandas dataframe we created above
    for row in df.values:
        x1, y1, x2, y2, _, class_id = row.astype(int)
        
        # to check if the face was already detected or not
        if class_id not in marked:

            # checking if the face is in the polygon zone or not
            if polygon.contains(Point(x1, y1)) and polygon.contains(Point(x2, y2)):
                face = frame[y1:y2, x1:x2] # face cordinates
                class_name = class_list[class_id] # name which we will check from class file which we importeed before the video loop started
                

                cv2.imshow(f'{class_name}', face)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # when detected face then update the attandence file
                file = open(f"attendance{current_day}.txt", 'a')
                file.write(f'{class_name}, Present, {current_time} \n')
                marked.add(class_id)
    
                # creating crop face saving file
                img = np.zeros((300, 200, 3), dtype = np.uint8)
                x_offset = 50
                y_offset = 50
                img[y_offset:y_offset+face.shape[0], x_offset:x_offset+face.shape[1]] = face

                # the name and present status puttext area
                font = cv2.FONT_HERSHEY_SIMPLEX
                l1 = 'Name:' + class_name
                l2 ='Status: Present'
                l3= 'Time:' + current_time
                text_color = (255, 255, 255)
                text_size1 = cv2.getTextSize(l1, font, 0.5, 1)[0]
                text_size2 = cv2.getTextSize(l2, font, 0.5, 1)[0]
                text_size3 = cv2.getTextSize(l3, font, 0.5, 1)[0]
                text_pos = (10, 150)

                # add text to blank image
                cv2.putText(img, l1, (x_offset, y_offset+face.shape[0]+text_size1[1]), font, 0.5, text_color, 1)
                cv2.putText(img, l2, (x_offset, y_offset+face.shape[0]+text_size1[1]+text_size2[1]+5), font, 0.5, text_color, 1)
                cv2.putText(img, l3, (x_offset, y_offset+face.shape[0]+text_size1[1]+text_size2[1]+text_size3[1]+15), font, 0.5, text_color, 1)

                # saving the new image to our specified directory
                file_name= '{}.jpg'.format(class_name)
                file_path= './cropped_images/' + file_name
                cv2.imwrite(file_path, img)

    cv2.rectangle(frame, rectangle_st, rectangle_ed, (255,0,0), 2)
    

    frame= cv2.resize(frame,(1020,500))
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


# realsing and destroying window
cap.release()
cv2.destroyAllWindows()
