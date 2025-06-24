'''
Code to split the dataset
import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Paths
images_dir = 'YOLO_Dataset/images'
labels_dir = 'YOLO_Dataset/labels'

output_images_train = 'YOLO_Dataset/images/train'
output_images_val = 'YOLO_Dataset/images/val'
output_labels_train = 'YOLO_Dataset/labels/train'
output_labels_val = 'YOLO_Dataset/labels/val'

# Create output dirs
for d in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
    os.makedirs(d, exist_ok=True)

# Gather all label files and their corresponding image
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
class_to_files = defaultdict(list)

# Group by class (based on the first class ID found in each file)
for label_file in label_files:
    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            continue
        classes = set([int(line.strip().split()[0]) for line in lines])
        # Assign to each class (multi-label will duplicate in multiple buckets)
        for cls in classes:
            class_to_files[cls].append(label_file)

# Merge files from all classes, deduplicate
all_files = set()
for file_list in class_to_files.values():
    all_files.update(file_list)

# Convert to list
all_files = list(all_files)

# Split balanced by filename (not perfect stratified but helps keep variation)
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Helper to copy files
def copy_files(files, img_dst, label_dst):
    for label_file in files:
        img_file = label_file.replace('.txt', '.jpg')  # or .png if you use that
        src_img = os.path.join(images_dir, img_file)
        src_label = os.path.join(labels_dir, label_file)
        if os.path.exists(src_img) and os.path.exists(src_label):
            shutil.copy(src_img, os.path.join(img_dst, img_file))
            shutil.copy(src_label, os.path.join(label_dst, label_file))

# Copy to train/val
copy_files(train_files, output_images_train, output_labels_train)
copy_files(val_files, output_images_val, output_labels_val)

print(f"✅ Done! Train: {len(train_files)}, Val: {len(val_files)}")
'''


'''
# Code to train the model

from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # nano version for fast training/testing

# Train the model
model.train(
    data='YOLO_Dataset/data.yaml',   # Path to your data.yaml
    epochs=50,
    imgsz=640,
    batch=16,
    name='waste-bin-detector',
    project='trash_yolo_project',
)
'''


'''
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8s.pt')  # nano version for fast training/testing

# Train the model
model.train(
    data='YOLO_Dataset/data.yaml',   # Path to your data.yaml
    epochs=50,
    imgsz=640,
    batch=16,
    name='waste-bin-detector-v8s-d',
    project='trash_yolo_project',
)
'''



from ultralytics import YOLO
import cv2, time

# Define what can go where
bin_rules = {
    "Biomüll": ["Food scraps", "Eggshells", "Tea bags"],
    "Papier":  ["Newspapers", "Cardboard", "Envelopes"],
    "Restmüll": ["Cigarette butts", "Vacuum cleaner bags", "Hygiene products"],
    "Glas": ["Bottles", "Jars (without lids)"]
}

# Load your model
model = YOLO('trash_yolo_project/waste-bin-detector-v8s/weights/best.pt')

# Camera stream (change IP as needed)
stream_url = 'http://192.168.0.109:4747/video'

CONF = 0.65
IOU = 0.35
SIZE = 640

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ Stream lost"); break

    t0 = time.time()

    # Run prediction
    results = model.predict(frame, imgsz=SIZE, conf=CONF, iou=IOU, stream=False)[0]

    # Copy the annotated frame
    annotated = results.plot().copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
        info = bin_rules.get(cls_name, ["No info available"])
        text = f"{cls_name}: " + ", ".join(info[:2])  # Show up to 2 items

        # Draw text
        # Calculate text size
        (font_w, font_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)

        # Coordinates for the background rectangle
        rect_x1 = x1
        rect_y1 = y1 + 5
        rect_x2 = x1 + font_w + 10
        rect_y2 = y1 + font_h + 20

        # Ensure the box doesn't go out of frame
        rect_x2 = min(rect_x2, annotated.shape[1] - 1)
        rect_y2 = min(rect_y2, annotated.shape[0] - 1)

        # Draw background rectangle (filled with opacity)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        alpha = 0.5
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

        # Draw the text over the rectangle
        cv2.putText(annotated, text, (x1 + 5, y1 + font_h + 5),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Show FPS
    fps = 1.0 / (time.time() - t0)
    cv2.putText(annotated, f"{fps:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display result
    cv2.imshow("Trash Bin Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

