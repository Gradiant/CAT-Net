import json
from pathlib import Path

import fire
from loguru import logger
from tqdm import tqdm


@logger.catch(reraise=True)  # noqa: C901
def split_from_annots(
    annotations_file: str,
    folders_train: [],
    folders_test: [],
    folders_val: [],
    output_train_file: str,
    output_test_file: str,
    output_val_file: str,
) -> None:
    """Split based on matching each image's `property_name` with `pattern_to_match`.

    Args:
        annotations_file:
            File in COCO Classification/Detection/Segmentation format.
        folders_train:
            Array with folders names where train data is stored.
        folders_test:
            Array with folders names where test data is stored.
        folders_val:
            Array with folders names where val data is stored.
        output_train_file:
            File where split of train images will be saved.
        output_test_file:
            File where split of test images will be saved.
        output_val_file:
            File where split of validation images will be saved.

    """
    Path(output_train_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_test_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_val_file).parent.mkdir(parents=True, exist_ok=True)

    class Splits:
        data_train = {}
        data_test = {}
        data_val = {}
        im_ids_train = []
        im_ids_test = []
        im_ids_val = []
        folders_train_split = folders_train
        folders_test_split = folders_test
        folders_val_split = folders_val


    list_splits = ["train","test","val"]
    splits = Splits()

    logger.info(f"Loading annotations from {annotations_file} ...")
    with open(annotations_file) as anns:
        data = json.load(anns)

        for split in list_splits:   #initialiation
            # setattr(splits, f"data_{split}", {})
            getattr(splits, f"data_{split}")["categories"]= data["categories"]
            getattr(splits, f"data_{split}")["images"]= []
            getattr(splits, f"data_{split}")["annotations"]= []
            setattr(splits, f"im_ids_{split}", [])

        # split images data
        logger.info("Split images data...")
        for images in data["images"]:
            for split in list_splits: 
                split_folders = getattr(splits, f"folders_{split}_split")
                for folder in split_folders:
                    if folder in images["file_name"].split('/'):
                        getattr(splits, f"im_ids_{split}").append(images["id"])
                        getattr(splits, f"data_{split}")['images'].append(images)

        for split in list_splits:
            logger.info(f"There are {len(getattr(splits, f'im_ids_{split}'))} imagenes in split {split}")


        pbar = tqdm(total=len(data["annotations"]))
        logger.info("Split annotations data...")

        set_dict = {split:set(getattr(splits, f"im_ids_{split}")) for split in list_splits}
        
        for i, annots in enumerate(data["annotations"]):
            pbar.update(1)
            for k, v in set_dict.items():
                if annots["image_id"] in v:
                    getattr(splits, f"data_{k}")['annotations'].append(annots)
        pbar.close()


    data_dict = {
        output_train_file: splits.data_train,
        output_test_file: splits.data_test,
        output_val_file: splits.data_val,
    }

    for output_file, data in data_dict.items():
        logger.info(f"Writing {output_file}...")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Done!")


if __name__ == "__main__":
    fire.Fire(split_from_annots)