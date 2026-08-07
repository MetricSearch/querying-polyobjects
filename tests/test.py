from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from scipy.io import loadmat
from PIL import Image
import requests
from io import BytesIO
from  polyhood import Hood
import os
from tqdm import tqdm
import pickle

IMAGE_FILES_PER_FOLDER = 10000

# Loading data
print("Loading data")
hood_instance_gmsed = Hood(source="../results_sub_image/db_results/db_gmsed.zarr", multi=True)
hood_instance_gmsed.make_ann_index("KNN","MSED")

hood_instance_euc = Hood(source="../results_sub_image/db_results/db_full_img.zarr", multi=False)
hood_instance_euc.make_ann_index("KNN","euclidean")

# Helper functions
def get_mf_image(index:int, size:Tuple[int,int]= None) -> Image.Image:

    folder_idx = index // IMAGE_FILES_PER_FOLDER
    url = f"https://similarity.cs.st-andrews.ac.uk/mirflickr/images/{folder_idx}/{index}.jpg"

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    if size:

        assert len(size) ==  2, "Size tuple must have two elements."
        img = img.resize(size)

    return img

def make_mf_image_grid(img_indices: np.array, num_cols:int, num_rows:int, img_w:int, img_h:int) -> Image.Image:

    images = [get_mf_image(i)for i in img_indices]

    grid = Image.new("RGB", size=(num_cols* img_w, num_rows* img_h))

    for idx, img in enumerate(images):

        img = img.resize((img_w, img_h), Image.Resampling.BILINEAR)

        grid.paste(img, box=(idx% num_cols * img_w, idx // num_cols * img_h))

    return grid

def load_encodings(encodings_dir:Path, key:str = 'features')-> np.array:

    paths = encodings_dir.glob("*.mat")

    paths = sorted(paths, key=lambda p: int(p.stem))
    encodings = [loadmat(p)[key] for p in paths]
    encodings = np.concatenate(encodings)

    return encodings 

def load_id_of_image(hood_instance, index):
    db_ind = -1
    for i, m in enumerate(hood_instance.store.metadatas):
        if (eval(m)['path'] == "distribution_results/results/"+str(index)+"/"+str(index)+".jpg"):
            db_ind = i
    return db_ind

def load_imgs_from_id(elements):
    img_ids = [int(os.path.basename(eval(metadata)['path']).removesuffix('.jpg')) for id, embedding, metadata, distance in elements]
    return img_ids


def perform_search_comparison(id):
    # IMG_INDEX = load_id_of_image(hood_instance_gmsed, id)
    IMG_INDEX = id

    id, embedding, metadatas = hood_instance_gmsed.get([IMG_INDEX])[0]
    id_euc, embedding_euc, metadatas_euc = hood_instance_euc.get([IMG_INDEX])[0]
    id, embedding, metadatas = hood_instance_imsed.get([IMG_INDEX])[0]

    # print("Searching MSED")
    results_msed = hood_instance_gmsed.query([embedding], no_neigh=24)
    # print("Searching Euclidean")
    results_euc = hood_instance_euc.query([embedding_euc], no_neigh=24)

    return [results_msed, results_euc]

img_ids = [393054,208214,576659,298727,379609,44322,547630,97855,594274,487768,341454,146620,456865,588089,521740,19371,422105,320213,284212,289264,460817,477154,73918,291174,336630,522754,419389,430173,39009,472422,171177,124032,239046,318237,423956,220139,194664,349475,525663,408901,39265,200012,525020,61386,635907,348369,567845,287911,610819,59050,337744,234317,71578,553199,474518,644376,483570,315592,9083,325548,48097,324900,184479,458027,104029,101812,134819,146730,307272,113471,336563,37702,294628,349135,537362,524792,490340,209970,79525,458823,30073,360587,360444,192513,622163,241009,326975,524274,489089,531592,241136,145829,251288,231124,391958,630714,99252,81009,40769,395267,630253,134249,541057,76114,332218,624914,323065,389289,88282,188362,404442,46174,567918,259241,515063,635159,571358,231645,14039,173800,308482,462865,188735,87051,160028,390761,16437,132562,601608,517656,554042,29755,355876,527071,572773,463190,374854,511498,528548,335118,305833,45351,647209,351753,168320,264382,574225,371188,260690,460020,75037,367402,646639,604513,433529,536844,74039,619902,569997,206140,164142,256942,438453,219996,347499,38870,199073,640722,217238,18374,304284,331193,468552,567612,434521,284469,116401,35391,233945,110031,171441,167101,635278,482330,580520,438437,536075,644538,344331,151176,484005,377878,568897,316119,468833,127553,186373,634850,240326,255189,86777,251742,145924,396387,437512,422286,389628,636500,116357,532130,112332,10307,550853,622077,150783,543929,488765,192294,141709,159514,297123,389979,622751,401689,7635,575504,530728,404852,475298,15139,375734,241806,484096,586201,96481,542284,271351,53494,185686,98788,189667,228898,88139,89833,251835,634715,529589,130997,77436,402653,460983,317662,318587,75614,494348,259054,616780,575527,192616,577221,187388,259106,443747,7984,14638,165966,133200,638408,614196,7983,467951,182541,118875,119234,114808,593696,32328,527732,225811,186108,406080,134823,165930,186874,541966,25678,60640,339650,200099,520240,64507,442698,197297,315378,334073,572857,562655,310596,235463,602844,430916,532569,551130,172416,32658,100365,134826,568479,365574,238839,247954,92872,601801,259675,511770,409033,584468,403572,248006,14242,406139,326473,4701,92333,634543,134723,370943,280538,481960]
full_results = []
for index in tqdm(img_ids, total=len(img_ids), desc="Searching for nearest neighbours", unit=" Queries "):
    # print ("Comparing for: " + str(index))
    nearest_neigbours = perform_search_comparison(index)
    full_results.append(nearest_neigbours)

with open('file.pkl', 'wb') as file:
    pickle.dump(full_results, file)