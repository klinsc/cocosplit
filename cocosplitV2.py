# read coco_annotations.json of Coco dataset and split it into train and test sets, using StratifiedShuffleSplit

import argparse
import json
import os

import funcy
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Splits COCO annotations file into training and test sets."
    )
    parser.add_argument(
        "annotations",
        metavar="coco_annotations",
        type=str,
        help="Path to COCO annotations file.",
    )
    parser.add_argument("train", type=str, help="Where to store COCO training annotations", default='train.json')
    parser.add_argument("test", type=str, help="Where to store COCO test annotations", default='test.json')
    parser.add_argument(
        "--having-annotations",
        dest="having_annotations",
        action="store_true",
        help="Ignore all images without annotations. Keep only these with at least one annotation",
    )
    parser.add_argument(
        "-s",
        dest="split",
        type=float,
        required=True,
        help="A percentage of a split; a number in (0, 1)",
    )

    return parser.parse_args()


def save_coco(file, images, annotations, categories):
    with open(file, "wt", encoding="UTF-8") as coco:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            coco,
            indent=2,
            sort_keys=True,
        )


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    return funcy.lfilter(lambda a: int(a["image_id"]) in image_ids, annotations)


def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i["image_id"]), annotations)

    return funcy.lfilter(lambda a: int(a["id"]) in annotation_ids, images)


def main():
    args = parse_args()

    with open(args.annotations, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    if args.having_annotations:
        images = filter_images(images, annotations)
        annotations = filter_annotations(annotations, images)

    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    image_ids = np.array(image_ids)

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.split)
    train_idx, test_idx = next(splitter.split(image_ids, image_ids))

    train_images = funcy.lremove(lambda i: int(i["id"]) not in image_ids[train_idx], images)
    test_images = funcy.lremove(lambda i: int(i["id"]) not in image_ids[test_idx], images)

    train_annotations = filter_annotations(annotations, train_images)
    test_annotations = filter_annotations(annotations, test_images)

    save_coco(args.train, train_images, train_annotations, categories)
    save_coco(args.test, test_images, test_annotations, categories)


if __name__ == "__main__":
    main()