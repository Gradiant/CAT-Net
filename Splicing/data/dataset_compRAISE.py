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


class compRAISE(AbstractDataset):
    """
    directory structure
    compRAISE (dataset_path["compRAISE"] in project_config.py)
    ├── cm_images
    ├── cm_masks
    └── sp_images ...
    """
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/compRAISE.txt"
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['compRAISE']
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip() for t in f.readlines()]

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index]
        return self._create_tensor(tamp_path, mask=None)

    def get_qtable(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index]
        if not str(tamp_path).lower().endswith('.jpg'):
            return None
        DCT_coef, qtables = self._get_jpeg_info(str(tamp_path))
        Y_qtable = qtables[0]
        return Y_qtable
