from sub_image_classifier.encoder import Encode


## Normal folder path
#FOLDER_PATH =  "../random-batch-of-images/mirflickr/"
## Dada folder path
FOLDER_PATH =  "../../experiments/distribution_results/results/"
OUT_PATH =  "../../experiments/encode_results/raw_results/"
## Normal image analysis

encoder = Encode.Encode()
encoder.encodes(FOLDER_PATH,OUT_PATH,dirs=False,sub_mode=False)