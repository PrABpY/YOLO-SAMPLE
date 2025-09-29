import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
for cid, name in model.names.items():
    print(cid, name)

wanted_names = ["book", "bottle","cup", "umbrella", "clock"]
name2id = {name: cid for cid, name in model.names.items()}
classes = [name2id[n] for n in wanted_names if n in name2id]

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok : break

    results = model.predict(frame, conf=0.3
                           #  , classes=classes
                            , verbose=False)
    annotated = results[0].plot()   

    cv2.imshow("YOLOv8 (PT)", annotated)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
