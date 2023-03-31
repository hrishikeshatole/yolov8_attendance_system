from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8m.pt")
    model.train(data="data_custom.yaml", batch=4, imgsz=640, epochs=50, workers=1)
