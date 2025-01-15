import cv2
import math
import cvzone
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Class labels
class_labels = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage',
    'Taillight-Damage', 'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent',
    'front-bumper-dent', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent',
    'Scratches', 'Paint-Peel-Off', 'Broken-Grill', 'Tyre-Damage',
    'Side-Indicator-Damage', 'Fog-Light-Damage', 'Radiator-Damage', 'Engine-Cover-Damage',
    'Door-Handle-Damage', 'Exhaust-Damage', 'Wheel-Rim-Damage'
]

# Estimated insurance cost for each damage type
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

# Load the image
image_path = r"C:\Users\DEEPIKA\Downloads\damage car2.jpg"
img = cv2.imread(image_path)

# Run YOLO model
results = yolo_model(img)

total_insurance_amount = 0

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.3:
            # Draw bounding box and label
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

            # Add insurance cost for detected damage
            damage_type = class_labels[cls]
            total_insurance_amount += insurance_costs.get(damage_type, 0)

# Print total insurance amount in the output window
print(f'Total Insurance Amount: Rs. {total_insurance_amount}')

# Show the image
cv2.imshow("Image", img)

# Close window when 'q' button is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
