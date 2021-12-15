"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
27 Jan 2021

This file is deprecated. Use tamp_COCO instead.
"""
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch


class DOCIMANv1(AbstractDataset):
    """
    directory structure
    tampCOCO (dataset_path["tampCOCO"] in project_config.py)
    ├── cm_images
    ├── cm_masks
    └── sp_images ...
    """
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str, istrain: bool):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "cm_COCO_train_list.txt"
        :param istrain: True for load train paths or false to load validation paths
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        # self._root_path = project_config.dataset_paths['COCOannot']

        self._images_path = Path(project_config.dataset_paths['DOCIMANv1_images'])
        if istrain:
            self._masks_path = Path(project_config.dataset_paths['DOCIMANv1_masks_train'])
        else:
            self._masks_path = Path(project_config.dataset_paths['DOCIMANv1_masks_val'])
        self._list_path = project_config.dataset_paths['DOCIMANv1_list']

        #read split list
        #tamp_list generated with 
        with open(Path(self._list_path) / tamp_list, "r") as f:
            self.tamp_list = [t.strip() for t in f.readlines()]

    def get_tamp(self, index):
        # open the mask image and pass the path to the input image
        # in our scripts maks are always in png format, and input images should be in .jpg format!

        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        # rootpathim = self._root_path / "images/"
        # rootpathmask =  Path("/media/BM/databases/DEFACTO/masks/")

        #chek extension of images in the images folder
        # if not(all(File.lower().endswith(".jpg") for File in os.listdir(self._images_path))):
        #     #TODO: Allow .jpeg extensions!
        #     raise ValueError('Input images should have .jpg extension.')

        maskfilename =  self.tamp_list[index] + ".png"
        # print("{} - {}".format(index, maskfilename))
        tamp_path = self._images_path / str(self.tamp_list[index] + ".jpg")
        mask_path = self._masks_path / maskfilename
        mask = np.array(Image.open(mask_path).convert('L'))
        mask[mask > 1] = 1
        return self._create_tensor(tamp_path, mask)

    def get_qtable(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._images_path / self.tamp_list[index] + ".jpg"
        # if not str(tamp_path).lower().endswith('.jpg'):
        #     #TODO: Allow .jpeg extensions!
        #     return None
        DCT_coef, qtables = self._get_jpeg_info(str(tamp_path))
        Y_qtable = qtables[0]
        return Y_qtable