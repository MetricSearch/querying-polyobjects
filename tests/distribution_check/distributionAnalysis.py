from sub_image_classifier import Classifier
import pandas as pd
import os
from tqdm import tqdm


NO_IMAGES = 10
## Normal folder path
#FOLDER_PATH =  "../random-batch-of-images/mirflickr/"
## Dada folder path
FOLDER_PATH =  "/Volumes/Data/mf/images/"
## Normal image analysis
# image_files = [os.path.join(FOLDER_PATH, f) for  f in os.listdir(FOLDER_PATH)
#                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

## Dada image analysis
image_files = []
j = 0
for root, dirs, files in os.walk(FOLDER_PATH):
    if j >= NO_IMAGES:
            break
    for f in files:
        if j >= NO_IMAGES:
            break
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            j+=1
            image_files.append(os.path.join(root, f))

classifier=Classifier()

analysis_data = []
box_data = []

for image_path in tqdm(image_files, desc="Processing images", unit="image"):
# for image_path in image_files:
    classifier.classify(image_path=image_path, save_cropped_images=True)
    boxes = classifier.results[0].boxes
    # Append data
    analysis_data.append({
        'image_name': os.path.basename(image_path),
        'num_detections': len(boxes)
    })

    for i, box in enumerate(boxes):
        # xywh: center_x, center_y, width, height
        x_center, y_center, width, height = box.xywh[0]
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        box_data.append({
            'image_name': os.path.basename(image_path),
            'box_index': i,
            'width': float(width),
            'height': float(height),
            'class': cls,
            'confidence': conf
        })

# Convert to DataFrame
df = pd.DataFrame(analysis_data)
df2 = pd.DataFrame(box_data)

# Save to CSV
df.to_csv('distribution_'+str(NO_IMAGES)+'_dada.csv', index=False)
df2.to_csv('boxes_distribution_'+str(NO_IMAGES)+'_dada.csv', index=False)