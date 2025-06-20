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
    name='waste-bin-detector-v8s',
    project='trash_yolo_project',
)
'''



# Testing the model
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('trash_yolo_project/waste-bin-detector-v8s/weights/best.pt')

# Replace with your phone's actual stream URL
stream_url = 'http://172.20.10.3:4747/video'

# Start video capture from IP stream
cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection on the frame
    results = model.predict(frame, conf=0.4, stream=False)

    # Plot results on frame
    annotated_frame = results[0].plot()

    # Show live stream with predictions
    cv2.imshow("Trash Bin Detection", annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

