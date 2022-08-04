"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020
"""
from tarfile import LENGTH_LINK
from sklearn.model_selection import train_test_split
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
from random import shuffle
from PIL import Image
from pathlib import Path
import mlflow

class DOCUMENTS(AbstractDataset):
    """
    directory structure:
    forgeries_documents (dataset_path["DOCUMENTS"] in project_config.py)
    ├── resized_images: 38478 images of text documents.
    │
    ├── forged_images:
        ├── Splicing: 871317 images and their masks with QF combinations from 60 to 100. 
        └── Copy-Move: 136014 images and their masks with QF combinations from 60 to 100, with step=5.
        
    
    """
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str, read_from_jpeg=False):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/forgeries_documents_train.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['DOCUMENTS']
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]
        self.read_from_jpeg = read_from_jpeg

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        mask_path = self._root_path / self.tamp_list[index][1]
        # Descomentar para guardar máscaras/imágenes originales en mlfow
        # imgname = str(tamp_path).split("/")[-1]
        # maskname = str(mask_path).split("/")[-1]
        # Path(mlflow.get_artifact_uri()[7:]+"/Masks/").mkdir(parents=True, exist_ok=True)
        # Path(mlflow.get_artifact_uri()[7:]+"/Images/").mkdir(parents=True, exist_ok=True)
        # Image.open(tamp_path).save(mlflow.get_artifact_uri()[7:]+"/Images/"+imgname)
        if self.tamp_list[index][1] == 'None':
            mask = None
        else:
            mask = np.array(Image.open(mask_path).convert("L"))
            # Image.open(mask_path).save(mlflow.get_artifact_uri()[7:]+"/Masks/"+maskname)
            mask[mask > 0] = 1

        return self._create_tensor(tamp_path, mask)
        
    def get_qtable(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        if not str(tamp_path).lower().endswith('.jpg'):
            return None
        DCT_coef, qtables = self._get_jpeg_info(tamp_path)
        Y_qtable = qtables[0]
        return Y_qtable


if __name__ == '__main__':
    root = project_config.dataset_paths['DOCUMENTS']
    splicing_root = root / "forged_images/Splicing"
    copy_move_root = root / "forged_images/Copy-Move"
    authentic_root = root / "resized_images"
    # Splicing
    # idx = 0
    # list_data, list_train, list_val = [], [], []

    # for file in os.scandir(splicing_root):
    #     if idx % 1000 == 0:
    #         print(idx)
    #     idx += 1
    #     if "mask" not in file.name:
    #         list_data.append(str(Path("forged_images/Splicing") / file.name))

    
    # for file in sorted(list_data):
    #     list_train.append(','.join([file, str(os.path.splitext(file)[0]+".mask.jpeg")]))

    # with open("DOCUMENTS_splicing_train.txt", "w") as f:
    #     f.write('\n'.join(list_train)+'\n')

    # #Copy-move
    # for file in os.scandir(copy_move_root):
    #     if idx % 1000 == 0:
    #         print(idx)
    #     idx += 1
    #     if "mask" not in file.name:
    #         list_data.append(str(Path("forged_images/Copy-Move") / file.name))

    # with open("DOCUMENTS_cm_train.txt", "w") as f:
    #     f.write('\n'.join(list_data)+'\n')
        
    # for file in sorted(list_data):
    #     if i >= int(0.9*num_total) and i % 40 == 0:
    #         list_val.append(','.join([file, str(os.path.splitext(file)[0]+".mask.jpeg")]))
    #     else:
    #         list_train.append(','.join([file, str(os.path.splitext(file)[0]+".mask.jpeg")]))
    #     print(file)


    # # # Authentic
    # idx = 0
    # list_data = []
    # len_dataset = len(os.listdir(authentic_root))
    # for file in os.scandir(authentic_root):
    #     if idx % 1000 == 0:
    #         print(idx)
    #     if file.name.lower().endswith(".jpeg"):
    #         list_data.append(','.join([str(Path("resized_images") / file.name), str(None)]))
    #     idx += 1
            
    # training_data, valid_data = train_test_split(list_data, test_size=0.1, shuffle=True)
    # valid_data, test_data = train_test_split(valid_data, test_size=0.5, shuffle=True)

    # with open("DOCUMENTS_auth_train.txt", "w") as f:
    #     f.write('\n'.join(training_data)+'\n')
    # with open("DOCUMENTS_auth_val.txt", "w") as f:
    #     f.write('\n'.join(valid_data)+'\n')
    # with open("DOCUMENTS_auth_test.txt", "w") as f:
    #     f.write('\n'.join(test_data)+'\n')
        

