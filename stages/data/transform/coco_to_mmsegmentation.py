import json
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def coco_to_mmsegmentation(
    annotations_file: str, output_annotations_file: str, output_masks_dir: str
):
    """Convert json in [segmentation format](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation) to txt in [mmsegmentation format](https://mmsegmentation.readthedocs.io/en/latest/tutorials/new_dataset.html#reorganize-dataset-to-existing-format).

    Args:
        annotations_file:
            path to json in [segmentation format](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation)
        output_annotations_file:
            path to write the txt in [mmsegmentation format](https://mmsegmentation.readthedocs.io/en/latest/tutorials/customize_datasets.html#customize-datasets-by-reorganizing-data)
        output_masks_dir:
            path where the masks generated from the annotations will be saved to.
            A single `{file_name}.png` mask will be generated for each image.
    """
    import cv2
    import numpy as np
    from pycocotools.coco import COCO

    Path(output_annotations_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_masks_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading annotations form {annotations_file}")
    annotations = json.load(open(annotations_file))

    logger.info(f"Saving annotations to {output_annotations_file}")
    with open(output_annotations_file, "w") as f:
        for image in annotations["images"]:
            filename = Path(image["file_name"]).parent / Path(image["file_name"]).stem
            f.write(str(filename))
            f.write("\n")

    logger.info(f"Saving masks to {output_masks_dir}")
    coco_annotations = COCO(annotations_file)
    for image_id, image_data in coco_annotations.imgs.items():

        filename = image_data["file_name"]

        anns_ids = coco_annotations.getAnnIds(imgIds=image_id)
        image_annotations = coco_annotations.loadAnns(anns_ids)

        logger.info(f"Creating output mask for {filename}")

        output_mask = np.zeros(
            (image_data["height"], image_data["width"]), dtype=np.uint8
        )
        logger.info("Empty mask created")

        black_list_masks = ["Tp_S_NRN_S_B_ind10002_ind10002_20010.jpg", "Tp_S_NRN_S_B_ind10002_ind10002_20010.jpg"]
        for image_annotation in image_annotations:
            category_id = image_annotation["category_id"]
            logger.info(f"category_id {category_id}")

            try:
                if filename in black_list_masks:
                    logger.info("Skipping")
                    category_mask = output_mask
                else:
                    category_mask = coco_annotations.annToMask(image_annotation)
                    logger.info("ann to mask")

            except Exception as e:
                logger.warning(e)
                logger.warning(f"Skipping {image_annotation}")
                continue
            logger.info("change values")
            category_mask *= category_id
            category_mask *= output_mask == 0
            output_mask += category_mask

        output_filename = Path(output_masks_dir) / Path(filename).with_suffix(".png")
        output_filename.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writting mask to {output_filename}")
        cv2.imwrite(str(output_filename), output_mask)


if __name__ == "__main__":
    fire.Fire(coco_to_mmsegmentation)