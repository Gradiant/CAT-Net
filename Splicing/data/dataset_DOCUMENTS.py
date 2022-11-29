"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020
"""
from tarfile import LENGTH_LINK
from sklearn.model_selection import train_test_split
import project_config
from Splicing.data.AbstractDataset import AbstractDataset
from collections import Counter
import os
import numpy as np
from random import shuffle
from PIL import Image
from pathlib import Path
from random import shuffle

class DOCUMENTS(AbstractDataset):
    """
    directory structure:
    forgeries_documents (dataset_path["DOCUMENTS"] in project_config.py)
    ├── resized_images: 38478 images of text documents.
    │
    ├── forged_images:
        └── Copy-Move-50-100: 203476 images and their masks with QF combinations from 50 to 100, with step=2.
        
    
    """
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str, read_from_jpeg=False):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/DOCUMENTS_train.txt"
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
        if self.tamp_list[index][1] == 'None':
            mask = None
        else:
            mask = np.array(Image.open(mask_path).convert("L"))
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
    documents_root = "forged_images/photos_documents_iphone_11"
    idx = 0
    list_data, list_train, list_val = [], [], []

    # with open("/media/data/workspace/rroman/CAT-Net/DOCUMENTS_photos.txt", "r") as f:
    #     tamp_list = [t.strip().split(',') for t in f.readlines()]

    # for image in tamp_list:
    #     if "q2_80" in image[0] or "q2_84" in image[0]:
    #         print(image[0])/
    #         list_data.append(','.join([image[0], image[1]]))

    for file in os.scandir(str(root)+"/"+documents_root):
        if idx % 1000 == 0:
            print(idx, file.name)
        idx += 1
        if "mask" not in file.name:
            list_data.append(','.join([str(Path(documents_root) / file.name), str(Path(documents_root) / Path(os.path.splitext(file.name)[0]+".mask.jpeg"))]))

    # data_train, data_val = train_test_split(list_data, test_size=0.15, shuffle=True)
    # for file in data_train:
    #     list_train.append(','.join([file, str(os.path.splitext(file)[0]+".mask.jpeg")]))
    
    # for file in data_val:
    #     list_val.append(','.join([file, str(os.path.splitext(file)[0]+".mask.jpeg")]))
    # shuffle(list_data)
    # for qf1 in range(50, 101, 2):
    #     for qf2 in range(50, 101, 2):
    #         count, count_train = 0, 0
    #         for file in list_data:
    #             if "q1_"+str(qf1)+"_q2_"+str(qf2) in file:
    #                 count += 1
    #         print(f"Qf1={qf1}, qf2={qf2}, count={count}")
    #         for file in list_data:
    #             if "q1_"+str(qf1)+"_q2_"+str(qf2) in file:
    #                 if count_train < int(0.9*count):
    #                     list_train.append(','.join([file, str(os.path.splitext(file)[0]+".mask.jpeg")]))
    #                     count_train += 1
    #                 else:
    #                     list_val.append(','.join([file, str(os.path.splitext(file)[0]+".mask.jpeg")]))


    # print(len(list_train), len(list_val))
    # shuffle(list_train)
    # shuffle(list_val)

    with open("DOCUMENTS_photos_iphone_11.txt", "w") as f:
        f.write('\n'.join(list_data)+'\n')
    # with open("DOCUMENTS_v3_50-100_val.txt", "w") as f:
    #     f.write('\n'.join(list_val)+'\n')

        

