from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11s.pt')

    results = model.train(
        data=r'Action Recognition.yolo26/data.yaml',
        epochs=20,
        imgsz=640,
        batch=8,        
        device=0,
        workers=2,      
        name='action_recognition_v1'
    )






