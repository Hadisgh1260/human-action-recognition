from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained model
    model = YOLO(r'C:\Users\ASUS\runs\detect\action_recognition_v1-4\weights\best.pt')

    # Run prediction on an image (change the path to your image)
    results = model.predict(
        source="C:\\Users\\ASUS\\Desktop\\images_082_jpg.rf.GNSaptniQRRWmG0fIglf.jpg",  # مسیر عکس خودت رو اینجا بگذار
        device=0,
        save=True,    # نتیجه رو با باکس‌ها ذخیره می‌کنه
        conf=0.25     # حداقل اطمینان برای نشون دادن یه تشخیص
    )

    print("Prediction saved! Check the runs/detect/predict folder.")