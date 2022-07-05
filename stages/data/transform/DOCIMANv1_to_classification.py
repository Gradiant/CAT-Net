import json
from pathlib import Path
from typing import List

import fire
from loguru import logger
from tqdm import tqdm
import project_config
import os
import random
from sklearn.model_selection import train_test_split

@logger.catch(reraise=True)  # noqa: C901
def DOCIMANv1_classification(
    annotations_file : str
) -> None:
    """Random Split with 80% of images for training, 20% validation, using images from all folders.
    Args:
        annotations_file:
            path to json in [segmentation format](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation)
    """
    # Classification annotations
    logger.info(f"Loading annotations form {annotations_file}")
    annotations = json.load(open(annotations_file))

    list_train, list_val = [], []

    for i, image in enumerate(annotations["images"]):
        if i % 100 == 0:
            print(i)
        filename = str(image["file_name"])
        label = str(image["category_id"])
        list_train.append(','.join((filename, label)))
        
    list_train, list_val = train_test_split(list_train, test_size=0.2)
    

    with open("Splicing/data/DOCIMANv1_train.txt", "w") as f:
        f.write('\n'.join(list_train)+'\n')
    with open("Splicing/data/DOCIMANv1_val.txt", "w") as f:
        f.write('\n'.join(list_val)+'\n')
    
    print(len(list_train)) 
    print(len(list_val)) 

if __name__ == "__main__":
    fire.Fire(DOCIMANv1_classification) 