from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from datetime import datetime
import os
import argparse

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="path to the YOLO model file")
parser.add_argument("--classes", type=str, required=True, help="path to the classes file")
parser.add_argument("--video1", type=str, required=True, help="path to the input video file")
parser.add_argument("--video2", type=str, required=True, help="path to the input video file")
args = parser.parse_args()

# model initialize
model = YOLO(args.model) # we are using our custom trained yolo model

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
cap1 = cv2.VideoCapture(args.video1)
cap2 = cv2.VideoCapture(args.video2)

# seting points for zone1
pnts1= [(400, 100), (550, 100), (550,352), (400, 352)]
polygon= Polygon(pnts1)

# seting points for zone2
pnts2= [(166, 31), (305, 31), (305,267), (166, 267)]
polygon2= Polygon(pnts2)

# blue rectangle drawen for frame1
rect1_st = (400, 100)
rect1_ed = (550, 352)

# blue rectangle drawwen for frame2
rect2_st= (166, 31)
rect2_ed= (305, 267)

# flag to check the class is crop only 1 time
marked= set()

# flag for 2nd frame
marked2= set()

# starting the video
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        # Check if either video has ended
        break

    # Resize frames 
    frame1 = cv2.resize(frame1, (640, 352))
    frame2 = cv2.resize(frame2, (640, 352))
    # Draw the rectangle on frame1
    cv2.rectangle(frame1, rect1_st, rect1_ed, (0,0,255), 2)

    # Draw the rectangle on frame2
    cv2.rectangle(frame2, rect2_st, rect2_ed, (0,0,255), 2)

    # for 1st video

    # getting the bounding box from tensor
    results1= model.predict(frame1)
    boxes= results1[0].boxes    
    a= boxes.data.cpu()

    # appending it in this dataframe for futhur use and representation   
    df= pd.DataFrame(a).astype("int")

    # loop will be getting the information of bounding box using the pandas dataframe we created above
    for row in df.values:
        x1,y1,x2,y2,_,class_id = row.astype(int)
        
        # to check if the face was already detected or not
        if class_id not in marked:

            # checking if the face is in the polygon zone or not
            if polygon.contains(Point(x1, y1)) and polygon.contains(Point(x2, y2)):
                face = frame1[y1:y2, x1:x2] # face cordinates
                class_name= class_list[class_id] # name which we will check from class file which we importeed before the video loop started
                

                cv2.imshow(f'{class_name}',face)
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # when detected face then update the attandence file
                file= open(f"attendance{current_day}.txt", 'a')
                file.write(f'{class_name}, Present, Entry time: {current_time} \n')
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
                l3= 'Entry:' + current_time
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


    # 2nd frame

    # getting the bounding box from tensor
    results2= model.predict(frame2)
    boxes2= results2[0].boxes    
    a2= boxes2.data.cpu()

    # appending it in this dataframe for futhur use and representation   
    df= pd.DataFrame(a2).astype("int")

    # loop will be getting the information of bounding box using the pandas dataframe we created above
    for row2 in df.values:
        x1,y1,x2,y2,_,class_id = row2.astype(int)
        
        # to check if the face was already detected or not
        if class_id not in marked2:

            # checking if the face is in the polygon zone or not
            if polygon2.contains(Point(x1, y1)) and polygon2.contains(Point(x2, y2)):
                face2 = frame2[y1:y2, x1:x2] # face cordinates
                class_name= class_list[class_id] # name which we will check from class file which we importeed before the video loop started
                

                cv2.imshow(f'{class_name}',face2)
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # when detected face then update the attandence file
                file1= open(f"exit{current_day}.txt", 'a')
                file1.write(f'{class_name}, Present, Exit time: {current_time} \n')
                marked2.add(class_id)
    
                # creating crop face saving file
                img = np.zeros((300, 200, 3), dtype = np.uint8)
                x_offset = 50
                y_offset = 50
                img[y_offset:y_offset+face2.shape[0], x_offset:x_offset+face2.shape[1]] = face2

                # the name and present status puttext area
                font = cv2.FONT_HERSHEY_SIMPLEX
                l1 = 'Name:' + class_name
                l2 ='Status: Present'
                l3= 'Exit:' + current_time
                text_color = (255, 255, 255)
                text_size1 = cv2.getTextSize(l1, font, 0.5, 1)[0]
                text_size2 = cv2.getTextSize(l2, font, 0.5, 1)[0]
                text_size3 = cv2.getTextSize(l3, font, 0.5, 1)[0]
                text_pos = (10, 150)

                # add text to blank image
                cv2.putText(img, l1, (x_offset, y_offset+face2.shape[0]+text_size1[1]), font, 0.5, text_color, 1)
                cv2.putText(img, l2, (x_offset, y_offset+face2.shape[0]+text_size1[1]+text_size2[1]+5), font, 0.5, text_color, 1)
                cv2.putText(img, l3, (x_offset, y_offset+face2.shape[0]+text_size1[1]+text_size2[1]+text_size3[1]+15), font, 0.5, text_color, 1)

                # saving the new image to our specified directory
                file_name= '{}2.jpg'.format(class_name)
                file_path= './cropped_images/' + file_name
                cv2.imwrite(file_path, img)


    # Display the frames in separate windows
    combined_frame = cv2.hconcat([frame1, frame2])
    cv2.imshow('2 Videos', combined_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break


# realsing and destroying window
cap1.release()
cap2.release()
cv2.destroyAllWindows()

