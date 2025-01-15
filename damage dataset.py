import os
import cv2
import math
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

class_labels = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage',
    'Taillight-Damage', 'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent',
    'front-bumper-dent', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent',
    'Scratches', 'Paint-Peel-Off', 'Broken-Grill', 'Tyre-Damage',
    'Side-Indicator-Damage', 'Fog-Light-Damage', 'Radiator-Damage', 'Engine-Cover-Damage',
    'Door-Handle-Damage', 'Exhaust-Damage', 'Wheel-Rim-Damage'
]

insurance_costs = {
    'Bodypanel-Dent': 5000,
    'Front-Windscreen-Damage': 8000,
    'Headlight-Damage': 3000,
    'Rear-windscreen-Damage': 7500,
    'RunningBoard-Dent': 4000,
    'Sidemirror-Damage': 2000,
    'Signlight-Damage': 1500,
    'Taillight-Damage': 2500,
    'bonnet-dent': 6000,
    'boot-dent': 5500,
    'doorouter-dent': 4500,
    'fender-dent': 3500,
    'front-bumper-dent': 7000,
    'pillar-dent': 4000,
    'quaterpanel-dent': 5000,
    'rear-bumper-dent': 6500,
    'roof-dent': 6000,
    'Scratches': 2000,
    'Paint-Peel-Off': 3000,
    'Broken-Grill': 4500,
    'Tyre-Damage': 5000,
    'Side-Indicator-Damage': 1500,
    'Fog-Light-Damage': 3500,
    'Radiator-Damage': 10000,
    'Engine-Cover-Damage': 7000,
    'Door-Handle-Damage': 2500,
    'Exhaust-Damage': 8000,
    'Wheel-Rim-Damage': 4000
}

def calculate_insurance(image_path, dataset_dir):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not load image: {image_path}")
        return
    results = yolo_model(img)
    total_insurance_amount = 0
    detected_labels = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.3:
                damage_type = class_labels[cls]
                detected_labels.append(damage_type)
                total_insurance_amount += insurance_costs.get(damage_type, 0)

    print(f"Detected damage types in {image_path}: {', '.join(detected_labels)}")
    print(f"Total Insurance Amount for {image_path}: Rs. {total_insurance_amount}")

    print("Verifying against dataset...")
    for dataset_file in os.listdir(dataset_dir):
        if dataset_file.endswith(('.jpg', '.jpeg', '.png')):
            dataset_image_path = os.path.join(dataset_dir, dataset_file)
            print(f"Processing dataset image: {dataset_image_path}")

            dataset_img = cv2.imread(dataset_image_path)
            if dataset_img is None:
                print(f"Could not load dataset image: {dataset_file}")
                continue

            dataset_results = yolo_model(dataset_img)
            dataset_total_insurance = 0
            dataset_detected_labels = []
            for r in dataset_results:
                dataset_boxes = r.boxes
                for box in dataset_boxes:
                    dataset_cls = int(box.cls[0])
                    dataset_damage_type = class_labels[dataset_cls]
                    insurance_amount = insurance_costs.get(dataset_damage_type, 0)
                    dataset_total_insurance += insurance_amount
                    dataset_detected_labels.append(dataset_damage_type)
                    print(f"Detected {dataset_damage_type} in dataset image. Insurance Amount: Rs. {insurance_amount}")
            print(f"Detected damage types in dataset image {dataset_image_path}: {', '.join(dataset_detected_labels)}")
            print(f"Total Insurance Amount for dataset image {dataset_image_path}: Rs. {dataset_total_insurance}")

image_path = r"C:\Users\DEEPIKA\Downloads\damage car.jpg"
dataset_dir = r"C:\Users\DEEPIKA\Downloads\car damage"
calculate_insurance(image_path, dataset_dir)
