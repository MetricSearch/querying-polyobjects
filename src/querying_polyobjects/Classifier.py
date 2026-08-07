from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import os
import shutil

class Classifier:
    
    MAX_DET = 5
    CONFIDENCE_INTERVAL = 0.35
    analysis_number_of_boxes = None
    analysis_per_boxes = None
    NO_IMAGES = 0
    TEST_DIR = None
    results = []
    # Dada images location
    # /Volumes/Data/mf/images/

    def __init__(self, test_dir=None):
        self.model = YOLO("yolo11x.pt");
        if test_dir != None:
            test_dir = test_dir.replace("/","")
            self.TEST_DIR = test_dir


    # Get all images from from the images directory and sub directories
    def get_images(self,folder_path):
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, f))
        return image_files

    def save_images(self, results):
        for result in results:
                # To bypass issues with yolo structuring itself
                save_dir= "results/"+os.path.basename(result.path).replace('.jpg', '')+"/"
                if self.TEST_DIR is not None:
                    save_dir= self.TEST_DIR + "/" + save_dir
                os.makedirs(save_dir, exist_ok=True)

                # save the cropped images
                result.save_crop(save_dir=save_dir, file_name=(os.path.basename(result.path).replace('.jpg', '')+"-im"))
                result.save_txt(txt_file=save_dir+"/det.txt")

                # Then store the OG image
                if len(result.boxes) > 0:
                    result.save(filename=(os.path.basename(result.path)))
                    # print(save_dir)
                    # print(os.path.basename(result.path))
                    full_dest = os.path.join(save_dir, os.path.basename(result.path))
                    shutil.move(os.path.basename(result.path), full_dest)

                # Then store the OG image
                # full_dest = os.path.join(save_dir, os.path.basename(result.path))
                # shutil.copy(result.path, full_dest)

    # Classify the images
    def classify(self, image_path, save_classified_labels=False, save_cropped_images=False, save_confidence_intervals=False, verbose=False, stream=False, batch=False):
        analysis_data = []
        box_data = []

        if (not batch):
            self.results = self.model.predict(source=image_path, conf=self.CONFIDENCE_INTERVAL, max_det=self.MAX_DET, save_txt=save_classified_labels, save_conf=save_confidence_intervals, verbose=verbose, stream=stream)
            self.run_distribution_analysis(self.results, analysis_data, box_data)
            # If we want the images saved (which we usually do)
            if (save_cropped_images):   
                self.save_images(self.results)
        else:
            image_files = self.get_images(image_path)
            self.NO_IMAGES = len(image_files)
            for image_pather in tqdm(image_files, desc="Processing images", unit="image"):
                sing_res = (self.model.predict(source=image_pather, conf=self.CONFIDENCE_INTERVAL, max_det=self.MAX_DET, save_txt=save_classified_labels, save_conf=save_confidence_intervals, verbose=verbose, stream=stream))
                self.run_distribution_analysis(sing_res, analysis_data, box_data)
                if (save_cropped_images):   
                    self.save_images(sing_res)
                # self.results.append(sing_res[0])

        # Save them
        self.analysis_number_of_boxes = pd.DataFrame(analysis_data)
        self.analysis_per_boxes = pd.DataFrame(box_data)
            

    # For analysis purposes (testing + useful in final thing)
    def run_distribution_analysis(self, results, analysis_data, box_data):
           
        for result in results:
            
            boxes = result.boxes

            # Detection analysis
            analysis_data.append({
            'image_name': result.path,
            'num_detections': len(boxes)
            })

            # Box (sub image) specific analysis
            for i, box in enumerate(boxes):
                # xywh: center_x, center_y, width, height
                x_center, y_center, width, height = box.xywh[0]
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                box_data.append({
                    'image_name': result.path,
                    'box_index': i,
                    'width': float(width),
                    'height': float(height),
                    'class': cls,
                    'confidence': conf
                })

        

    # Describer (implemented from distribution check folder + useful for final thing)
    def describe_classifier(self):
        # If no analysis is done do analysis
        try:
            print(self.analysis_number_of_boxes['num_detections'].describe())
            print(self.analysis_per_boxes['width'].describe())
            print(self.analysis_per_boxes['height'].describe())
            print(self.analysis_per_boxes['confidence'].describe())
        # catch when df1 is None
        except AttributeError:
            self.classify()
            describe_classifier()
        # catch when it hasn't even been defined
        except NameError:
            self.classify()
            describe_classifier()
        

    def save_analysis(self, path:str = 'distribution.csv'):
        # Save to CSV
        self.analysis_number_of_boxes.to_csv('distribution_'+str(self.NO_IMAGES)+'_dada.csv', index=False)
        self.analysis_per_boxes.to_csv('boxes_distribution_'+str(self.NO_IMAGES)+'_dada.csv', index=False)