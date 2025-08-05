from ultralytics import YOLO
import torch

def main():



    model = YOLO("yolo11s.pt")  


    results = model.train(data=r"C:\Users\user\Desktop\img2text\data.yaml",
                            epochs=300, 
                            imgsz=1280, 
                            batch=2,
                            hsv_h=0.0,
                            hsv_s=0.0,
                            hsv_v=0.0,
                            translate=0.0,
                            scale=0.0,
                            fliplr=0.0,
                            mosaic=0.0,
                            erasing=0.0,
                            auto_augment=None,
                            # epochs = 100,
                            # imgsz = 640,
                            # batch=8
                            )

if __name__ == '__main__':
    with torch.no_grad():
        torch.cuda.empty_cache()

    main()  
