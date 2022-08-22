import json
from pathlib import Path

import fire
from loguru import logger
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@logger.catch(reraise=True)  # noqa: C901
def histograms_from_predictions(
    path_predictions_double: str,
    path_predictions_single: str,
    path_results: str,

) -> None:
    """Path where the prediction maps of double compressed images were stored.

    Args:
        annotations_train_file:
            Path where the prediction maps of double compressed images were stored.
        annotations_test_file:
            Path where the prediction maps of single compressed images were stored.
        path_results:
            Path where the results will be saved.

    """

    ##read all images and save pixel predictions
    dirs = {'single': path_predictions_single, 'double': path_predictions_double}
    pixels = {'single':[], 'double':[]}

    for k, dir_preds in dirs.items():
        logger.info(f'Process data {k}...')
        onlyfiles = [os.path.join(path_predictions_double, f) for f in os.listdir(path_predictions_double) if os.path.isfile(os.path.join(path_predictions_double, f))]
        logger.info(f'{len(onlyfiles)} images were read in {dir_preds} path')
        tmp_arr = []
        for im_file in onlyfiles:
            pred = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
            tmp_arr.append(pred.ravel())
        pixels[k] = np.array(tmp_arr).ravel()
        # plt.hist(pixels[k], bins = 30)

        logger.info(f'Number of pixels in {k} is {len(pixels[k])}')
    
    sns.histplot(pixels['double'], bins=82, kde=True, label='double', color='skyblue', alpha=.5)
    plt.legend()
    plt.show()
    plt.savefig(path_results + '/histd.png')
    sns.histplot(pixels['single'], bins=82, kde=True, label='single', color='red', alpha=.5)
    plt.legend()
    plt.show()
    plt.savefig(path_results + '/hists.png')

    logger.info("Done!")


if __name__ == "__main__":
    fire.Fire(histograms_from_predictions)