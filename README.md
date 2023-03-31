# Attendance System using YOLOv8 Custom Object Detection Model

This program uses a custom trained YOLOv8 object detection model to detect faces of individuals from a video feed and mark their attendance. The attendance is saved in a text file with the date and time stamp. Cropped images of the individuals with their name, present status and timestamp are also saved for record-keeping purposes.
 
# Requirements
    • Python3
    • YOLOv8
    • OpenCV
    • Pandas
    • NumPy
    • Shapely
    
# Installation
    • Clone the repository
    • Install the required libraries: pip install -r requirements.txt
    • Download the custom trained YOLOv8 model and place it in the project directory.
    • Create a file named "classes.txt" in the project directory with the names of the individuals to be marked for attendance.
    
# How to use
    • Open terminal
    • Navigate to clone repository and use: python yolo_attendance.py --model yolov8_custom.pt --classes classes.txt --video1 /home/hrishi/Yolov8/Yolo/1.mp4 --video2 /home/hrishi/Yolov8/Yolo/2.mp4
    • Here: 
    1. filename = yolo_attendance.py
    2. model = name of custom train model
    3. classes = path of classes file (classes file have name of all the registered objects)
    4. video = path of video file
    • A video feed will open up.
    • The program will detect faces and mark attendance as individuals enter a designated zone (set by points defined in the code).
    • Attendance will be saved in a text file with the date and timestamp, and cropped images will be saved in the “cropped images” directory.
    
# Future Improvements
    • Making code work on multiple cctv camera feeds at same time.
