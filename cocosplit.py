# read coco_annotations.json of Coco dataset and split it into train and test sets, using StratifiedShuffleSplit
# also filter out classes that has only one sample, because it can't be split into the training and testing sets

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
    parser.add_argument(
        "train",
        type=str,
        help="Where to store COCO training annotations",
        default="train.json",
    )
    parser.add_argument(
        "val",
        type=str,
        help="Where to store COCO val annotations",
        default="val.json",
    )
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


def save_coco(root, file, images, annotations, categories):
    with open(root + file, "wt", encoding="UTF-8") as coco:
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


def filter_one_sample_classes(annotations, categories):
    """
    Filter out classes that has only one sample, because it can't be split into the training and testing sets
    """
    category_ids = funcy.lmap(lambda c: int(c["id"]), categories)
    annotations_per_category = {id: [] for id in category_ids}

    for annotation in annotations:
        annotations_per_category[int(annotation["category_id"])].append(annotation)

    annotations_per_category = {
        id: annotations
        for id, annotations in annotations_per_category.items()
        if len(annotations) > 1
    }

    category_ids = list(annotations_per_category.keys())
    annotations = funcy.lfilter(
        lambda a: int(a["category_id"]) in category_ids, annotations
    )
    categories = funcy.lfilter(lambda c: int(c["id"]) in category_ids, categories)

    return annotations, categories


def main():
    args = parse_args()

    with open(args.annotations, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    if args.having_annotations:
        annotations = filter_annotations(annotations, images)
        images = filter_images(images, annotations)

    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    category_ids = funcy.lmap(lambda c: int(c["id"]), categories)

    annotations_per_category = {id: [] for id in category_ids}
    for annotation in annotations:
        annotations_per_category[int(annotation["category_id"])].append(annotation)

    train_annotations = []
    test_annotations = []

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.split, random_state=0)
    for category_id, annotations in annotations_per_category.items():
        train_index, test_index = next(
            sss.split(np.zeros(len(annotations)), np.zeros(len(annotations)))
        )

        for index in train_index:
            train_annotations.append(annotations[index])

        for index in test_index:
            test_annotations.append(annotations[index])

    train_image_ids = list(
        set(funcy.lmap(lambda a: int(a["image_id"]), train_annotations))
    )
    test_image_ids = list(
        set(funcy.lmap(lambda a: int(a["image_id"]), test_annotations))
    )

    train_images = funcy.lremove(lambda i: int(i["id"]) not in train_image_ids, images)
    test_images = funcy.lremove(lambda i: int(i["id"]) not in test_image_ids, images)

    # root is parent directory of annotations file
    root = os.path.dirname(args.annotations)

    save_coco(root, args.train, train_images, train_annotations, categories)
    save_coco(root, args.val, test_images, test_annotations, categories)


if __name__ == "__main__":
    main()
