"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 July 7, 2020
"""
from pathlib import Path

project_root = Path(__file__).parent
dataset_root = Path(r"/media/BM/databases/")
dataset_paths = {
    # Specify where are the roots of the datasets.
    #'FR': dataset_root / "FantasticReality_v1",
    #'IMD': dataset_root / "IMD2020",
    'CASIA': dataset_root / "CASIA/CASIAv2",
    # 'NC16': dataset_root / "NC2016_Test",
    # 'Columbia': dataset_root / "Columbia Uncompressed Image Splicing Detection",
    # 'Carvalho': dataset_root / "tifs-database",
    # 'tampCOCO': dataset_root / "tampCOCO",
    'DEFACTO': dataset_root / "DEFACTO-12K",
    
    'COCOannot_images_train': dataset_root / "DEFACTO-84K/train_images/",
    'COCOannot_masks_train':"results/data/transform/coco_to_mmsegmentation-defacto/masks/train/",
    'COCOannot_images_val': dataset_root / "DEFACTO-12K/images/",
    'COCOannot_masks_val': "results/data/transform/coco_to_mmsegmentation-defacto/masks/val/",
    'COCOannot_list': "results/data/transform/coco_to_mmsegmentation-defacto/",
    'Park': "/media/VA/BM_tmp/databases/Park/",
    'DOCIMANv1': dataset_root / "dociman/DOCIMANv1/",
    'DOCIMANv2': dataset_root / "dociman/DOCIMANv2/",
    'DOCUMENTS': dataset_root / "forgeries_documents/",
    
    # 'compRAISE': dataset_root / "compRAISE",
    # 'COVERAGE': dataset_root / "COVERAGE",
    # 'CoMoFoD': dataset_root / "CoMoFoD_small_v2",
    # 'GRIP': dataset_root / "CMFDdb_grip",
    'SAVE_PRED': project_root / "output_pred"
}



