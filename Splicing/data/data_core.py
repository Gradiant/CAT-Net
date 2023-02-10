"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 8, 2020
"""


import torch
from torch.utils.data import Dataset
import random
from Splicing.data.dataset_DOCIMANv1 import DOCIMANv1
from Splicing.data.dataset_DOCIMANv2 import DOCIMANv2
from Splicing.data.dataset_DOCIMANv2_combined import DOCIMANv2Combined
from Splicing.data.dataset_DOCUMENTS import DOCUMENTS
from Splicing.data.dataset_DOCUMENTS_combined import DOCUMENTS_Combined

from Splicing.data.dataset_arbitrary import arbitrary
from Splicing.data.dataset_arbitrary_cls import arbitraryCls
from Splicing.data.dataset_arbitrary_combined import arbitraryCombined
class SplicingDataset(Dataset):
    def __init__(self, crop_size, grid_crop, blocks=('RGB',), mode="train", DCT_channels=3, read_from_jpeg=False, class_weight=None):
        self.dataset_list = []
        if mode == "train":
            # self.dataset_list.append(DOCIMANv2(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/DOCIMANv2_train_no_white_crops.txt")) 
            self.dataset_list.append(DOCUMENTS_Combined(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/DOCUMENTS_combined_train_forged.txt"))

        elif mode == "valid":
            # self.dataset_list.append(DOCIMANv2(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/DOCIMANv2_val_no_white_crops.txt"))
            self.dataset_list.append(DOCUMENTS_Combined(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/DOCUMENTS_combined_val_forged.txt"))

        elif mode == "arbitrary":
            self.dataset_list.append(DOCUMENTS(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/GIMP_images.txt"))
            # self.dataset_list.append(arbitrary(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/DOCUMENTS_v2_50-100_val.txt"))

        elif mode == "arbitraryCls":
            self.dataset_list.append(arbitraryCls(crop_size, grid_crop, blocks, DCT_channels, "/media/data/workspace/rroman/CAT-Net/images//*", read_from_jpeg=read_from_jpeg))
            # self.dataset_list.append(DOCIMANv2(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/DOCIMANv2_val_filtered_no_white_crops.txt"))
        else:
            raise KeyError("Invalid mode: " + mode)
        if class_weight is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weight)
        self.crop_size = crop_size
        self.grid_crip = grid_crop
        self.blocks = blocks
        self.mode = mode
        self.read_from_jpeg = read_from_jpeg
        self.smallest = 31115 # 62080 #70342 31115 1059479 #len(self.dataset_list) # smallest dataset size (IMD:1869)

    def shuffle(self):
        for dataset in self.dataset_list:
            random.shuffle(dataset.tamp_list)

    def get_PIL_image(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_PIL_Image(index)

    def get_filename(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_tamp_name(index)

    def __len__(self):
        if self.mode == 'train':
            # class-balanced sampling
            return self.smallest * len(self.dataset_list)
        else:
            return sum([len(lst) for lst in self.dataset_list])

    def __getitem__(self, index):
        if self.mode == 'train':
            # class-balanced sampling
            if index < self.smallest * len(self.dataset_list):
                return self.dataset_list[index//self.smallest].get_tamp(index%self.smallest)
            else:
                raise ValueError("Something wrong.")
        else:
            it = 0
            while True:
                if index >= len(self.dataset_list[it]):
                    index -= len(self.dataset_list[it])
                    it += 1
                    continue
                return self.dataset_list[it].get_tamp(index)

    def get_info(self):
        s = ""
        for ds in self.dataset_list:
            s += (str(ds)+'('+str(len(ds))+') ')
        s += '\n'
        s += f"crop_size={self.crop_size}, grid_crop={self.grid_crip}, blocks={self.blocks}, mode={self.mode}, read_from_jpeg={self.read_from_jpeg}, class_weight={self.class_weights}\n"
        return s





