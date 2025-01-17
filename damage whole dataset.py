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

def calculate_insurance(dataset_dir):
    output_dir = os.path.join(dataset_dir, "output_images")
    os.makedirs(output_dir, exist_ok=True)

    for folder in ["training", "validation"]:
        damage_path = os.path.join(dataset_dir, folder, "damage")
        if not os.path.exists(damage_path):
            print(f"Category folder not found: {damage_path}")
            continue

        print(f"Processing images in {damage_path}...")

        for file_name in os.listdir(damage_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(damage_path, file_name)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Could not load image: {image_path}")
                    continue

                
                results = yolo_model(img)
                detected_labels = []
                total_insurance_amount = 0

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  
                        conf = box.conf[0]  
                        cls = int(box.cls[0])  

                        
                        if cls < len(class_labels):
                            label = class_labels[cls]  
                            if conf > 0.3:  
                                detected_labels.append(label)
                                total_insurance_amount += insurance_costs.get(label, 0)
                               
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            print(f"Warning: Detected class index {cls} is out of range for class_labels.")

               
                output_path = os.path.join(output_dir, f"{folder}_damage_{file_name}")
                cv2.imwrite(output_path, img)

                print(f"Processed {image_path}:")
                print(f"  Detected damage types: {', '.join(detected_labels)}")
                print(f"  Total insurance amount: Rs. {total_insurance_amount}")
                print(f"  Saved processed image to {output_path}\n")


dataset_dir = r"C:\Users\DEEPIKA\Downloads\car damage"
calculate_insurance(dataset_dir)
