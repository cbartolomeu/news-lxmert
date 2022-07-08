import base64
import json
from os import path
import pandas as pd

import torch
from tqdm import tqdm

from model import Model


def handle_exception(ex, message):
    print(ex)
    print(message)


def base64_encode(tensor):
    tensor = torch.squeeze(tensor)
    return base64.b64encode(tensor.detach().numpy()).decode("utf-8")


def write_info(image_info, filename):
    feats = {
        "roi_features": base64_encode(image_info["roi_features"]),
        "boxes": base64_encode(image_info["boxes"]),
        "normalized_boxes": base64_encode(image_info["normalized_boxes"]),
        "obj_ids": base64_encode(image_info["obj_ids"]),
        "obj_probs": base64_encode(image_info["obj_probs"]),
        "attr_ids": base64_encode(image_info["attr_ids"]),
        "attr_probs": base64_encode(image_info["attr_probs"]),
        "n_regions": torch.squeeze(image_info["boxes"]).size(0),
    }

    json_string = json.dumps(feats)
    with open(filename, "w") as f:
        f.write(json_string)


def extract_dataset(index, dataset_dir, images_dir, features_dir):
    model = Model()

    df = pd.read_csv(path.join(dataset_dir, index), delimiter='\t')

    for idx in tqdm(range(len(df))):
        image_metadata = df.iloc[idx]

        image_hash = image_metadata["imgFile"].replace(".jpg", "")
        image_path = f"{images_dir}/{image_hash}.jpg"
        dst_filename = f"{features_dir}/{image_hash}.json"

        if not path.exists(dst_filename):
            try:
                image_info = model.process_image(image_path)
                write_info(image_info, dst_filename)
            except Exception as ex:
                handle_exception(ex,
                                 f"[ERROR] Reading image in {image_path} in newspiece with id {idx}")


def main():
    dataset_dir = "/user/data/c.bartolomeu/media-eval"
    images_dir = "/user/data/c.bartolomeu/media-eval/images"
    feats_filename = "/user/data/c.bartolomeu/media-eval/feats"

    index = "train_mediaeval.tsv"
    extract_dataset(index, dataset_dir, images_dir, feats_filename)

    index = "MediaEvalNewsImagesBatch04images.tsv"
    extract_dataset(index, dataset_dir, images_dir, feats_filename)


if __name__ == '__main__':
    main()
