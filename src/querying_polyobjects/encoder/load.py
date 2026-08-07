from pathlib import Path
from typing import Callable, Optional, Any
import os
from tqdm import tqdm
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, is_image_file
import imagesize

class UnlabelledImageFolder(VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, sub_mode=True, img_mode=True) -> None:

        super().__init__(root, transform=transform, target_transform=None)
        self.filepaths = []
        self.classifications = []
        sub_class_str = None
        root = Path(root)

        with os.scandir(root) as top:
            for folder in tqdm(top, "Reading Images From Folder", unit=" Images "):
                if not folder.is_dir():
                    continue

                with os.scandir(folder.path) as subs:
                    for sub in subs:
                        if img_mode:
                            og_img = os.path.join(folder.path,folder.name + ".jpg")

                            if sub.is_dir():
                                if not sub_mode and os.path.isfile(og_img):
                                    if og_img not in self.filepaths:
                                        self.filepaths.append(og_img)
                                
                                sub_classifications = []
                                sub_class_str = ""
                                with os.scandir(sub.path) as imgs:
                                    for img in imgs:
                                        if img.name.lower().endswith((".jpg")) :
                                            width, height = imagesize.get(img)
                                            # if width>160 and height>173: 
                                            if sub_mode:
                                                self.filepaths.append(img.path)
                                                self.classifications.append(sub.name)
                                            else:
                                                sub_classifications.append(sub.name)
                        else:
                            if sub.name.lower().endswith((".jpg")):
                                self.filepaths.append(sub.path)
                                self.classifications.append(folder.name)
                            elif sub.is_dir():
                                with os.scandir(sub.path) as frames:
                                    for frame in frames:
                                        if frame.name.lower().endswith((".jpg")):
                                            self.filepaths.append(frame.path)
                                            self.classifications.append(sub.name)


                            
                if not sub_mode and sub_class_str is not None:
                    sub_class_str = ",".join(sub_classifications)
                    self.classifications.append(sub_class_str)



    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Any:
        try:
            path = str(self.filepaths[index])
            sample = default_loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample
        except Exception as e:
            print(f"Error loading image at {path}: {e}")  # Catch other exceptions
            return None  # Handle additional exceptions if necessary
