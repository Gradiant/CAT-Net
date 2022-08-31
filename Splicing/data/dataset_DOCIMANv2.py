"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
Aug 4, 2020

This dataset is used for pretraining the DCT stream on double JPEG detection.
"""
import project_config
from Splicing.data import AbstractDatasetCls
from PIL import Image
import numpy as np

class DOCIMANv2(AbstractDatasetCls.AbstractDatasetCls):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/DOCIMANv2_train.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['DOCIMANv2']
        with open("/home/rroman/workspace/CAT-Net/"+tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = str(self._root_path)+'/'+self.tamp_list[index][0]
        return self._create_tensor(tamp_path, int(self.tamp_list[index][1]))

if __name__ == '__main__':
    # Filter qfs so that qf2>qf1
    list_data = []
    input_file = "/home/rroman/workspace/CAT-Net/Splicing/data/DOCIMANv2_train.txt"
    with open(input_file, "r") as f:
        tamp_list = [t.strip().split(',') for t in f.readlines()]

    for image in tamp_list:
        if 'q2' in image[0]:
            qf1 = int(image[0].split('_')[-3])
            qf2 = int(image[0].split("_")[-1].split(".")[0])
            if qf2 > qf1:
                list_data.append(','.join((image[0], image[1])))
        else:
            list_data.append(','.join((image[0], image[1])))

    output_file = input_file.split(".")[0]+"_filtered.txt"
    with open(output_file, "w") as f:
        f.write('\n'.join(list_data)+'\n')
    print(len(list_data))
    