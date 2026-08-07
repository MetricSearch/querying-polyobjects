import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import zarr
from pathlib import Path
from typing import List
import numpy as np
import multiprocessing as mp
from sub_image_classifier.encoder.load import UnlabelledImageFolder
import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
import numcodecs


class Encode:

    model = None
    preprocess = None

    def __init__(self):
        self.model, self.preprocess = self.load_dino2l()

    def get_image_net_preprocessor(self):
        return Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    def load_dino2l(self) -> nn.Module:
        dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        preprocess = self.get_image_net_preprocessor()
        return dinov2_vits14, preprocess

    def save_features(self,arr: np.array, path: Path, 
                  image_filepaths: List[Path] = None, image_class_arr: List[str] = None) -> None:
        store = zarr.DirectoryStore(path)
        root = zarr.open_group(store=store, mode='w')
        N, D = arr.shape
        dset = root.create_dataset('features', shape=arr.shape, chunks=(min(10000, N), D), dtype=arr.dtype)
        dset[:] = arr

        # add the image paths
        if image_filepaths:
            image_filepaths_arr = np.array(image_filepaths, dtype=object)
            image_class_arr = np.array(image_class_arr, dtype=str)
            root.create_dataset('filenames', data=image_filepaths_arr, object_codec=numcodecs.JSON(), chunks=(arr.shape[0],), overwrite=True)
            root.create_dataset('classifications', data=image_class_arr, chunks=(arr.shape[0],), overwrite=True)

    def custom_collate_fn(self,batch):
        # Filter out None values from the batch
        batch = [item for item in batch if item is not None]
        return default_collate(batch)

    def encode_images(self, input_dir: Path, batch_size: 
                  int, device: str, num_workers: int, sub_mode=True,img_mode=True) -> np.array:
        dataset = UnlabelledImageFolder(input_dir, self.preprocess, sub_mode=sub_mode, img_mode=img_mode)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=self.custom_collate_fn)
        all_features = []
        with torch.no_grad():
             all_features = [self.model(xs.to(device)).detach().cpu().numpy() for xs in tqdm(loader)]
        return np.concatenate(all_features), dataset.filepaths, dataset.classifications

    def encodes(self, input_dir, output_path, batch_size=500, dirs=False, include_filenames=True, sub_mode=True, img_mode=True):

        input_dir = input_dir.lstrip("/")
        # print(input_dir, output_path, batch_size, dirs, format)

        output_path = output_path.lstrip("/")

        Path(output_path).mkdir(parents=True, exist_ok=True)

        # find all the directories to look for images in
        if dirs:
            image_dirs = [f for f in Path(input_dir).iterdir() if f.is_dir()]
            image_dirs = sorted(image_dirs, key=lambda p: p.name)
        else:
            image_dirs = [Path(input_dir)]

        # setup the pytorch device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        self.model.eval()  # set the model into evaluation mode

        # "huristic" to compute the number of loader workers
        num_workers = max(mp.cpu_count() // 2, 1)

        # iterate over the input dirs, encoding and outputting to disk
        for image_dir in image_dirs:
            features, image_filepaths, image_classes = self.encode_images(image_dir, batch_size, device, num_workers, sub_mode=sub_mode, img_mode=img_mode)
            filepath = Path(output_path, image_dir.stem).with_suffix(".zarr")
            image_filepaths = image_filepaths if include_filenames else None
            self.save_features(features, filepath, image_filepaths=image_filepaths, image_class_arr=image_classes)