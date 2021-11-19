import json
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)  # noqa: C901
def split_from_file(
    annotations_train_file: str,
    annotations_test_file: str,
    annotations_val_file: str,
    output_train_file: str,
    output_test_file: str,
    output_val_file: str,
) -> None:
    """Split based on matching each image's `property_name` with `pattern_to_match`.

    Args:
        annotations_train_file:
            File in COCO Classification/Detection/Segmentation format.
        annotations_test_file:
            File in COCO Classification/Detection/Segmentation format.
        annotations_val_file:
            File in COCO Classification/Detection/Segmentation format.
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

    logger.info(f"Loading annotations from {annotations_train_file} ...")
    with open(annotations_train_file) as anns:
        data_train = json.load(anns)

    with open(annotations_test_file) as anns:
        data_test = json.load(anns)

    with open(annotations_val_file) as anns:
        data_val = json.load(anns)

    data_dict = {
        output_train_file: data_train,
        output_test_file: data_test,
        output_val_file: data_val,
    }

    for output_file, data in data_dict.items():
        logger.info(f"Writing {output_file}...")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Done!")


if __name__ == "__main__":
    fire.Fire(split_from_file)