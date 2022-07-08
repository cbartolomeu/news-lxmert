import base64
import json
import pickle
from os import path

import torch
from tqdm import tqdm

from model import Model


def handle_exception(ex, message):
    print(ex)
    print(message)


def base64_encode(tensor):
    tensor = torch.squeeze(tensor)
    return base64.b64encode(tensor.detach().numpy()).decode("utf-8")


def get_image(news_filename, image_idx):
    try:
        # Read news piece
        news_piece = None
        with open(news_filename) as reader:
            news_piece = json.load(reader)

        assert (news_piece is not None)

        # Access image positions in sections
        image_positions = news_piece["image_positions"]
        assert (len(image_positions) > 0)

        image_position = image_positions[image_idx]

        # Get array of sections (paragraphs, captions, etc)
        parsed_sections = news_piece["parsed_section"]
        assert (len(parsed_sections) > image_position)

        # Obtain image section
        image_section = parsed_sections[image_position]
        assert (image_section["type"] == "caption")

        return image_section["hash"]
    except Exception as ex:
        handle_exception(ex,
                         f"[ERROR] Error processing newspiece in {news_filename} with image in position {image_idx}")


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
    images_index = pickle.load(open(index, "rb"))

    for idx in tqdm(range(len(images_index))):
        image_metadata = images_index[idx]
        news_id = image_metadata[0]
        image_position = image_metadata[1]

        news_filename = f"{dataset_dir}/{news_id}"

        image_hash = get_image(news_filename, image_position)
        image_path = f"{images_dir}/{image_hash}.jpg"
        dst_filename = f"{features_dir}/{image_hash}.json"

        if not path.exists(dst_filename):
            try:
                image_info = model.process_image(image_path)
                write_info(image_info, dst_filename)
            except Exception as ex:
                handle_exception(ex,
                                 f"[ERROR] Reading image in {image_path} in newspiece with id {image_metadata[0]}")


def main():
    index = "../datasets/nytimes/data/train-images.pickle"
    dataset_dir = "../datasets/nytimes/data/train"
    images_dir = "/storagebk/datasets/ny800k_goodnews/data/nytimes/images"
    feats_filename = "/user/data/c.bartolomeu/nytimes/feats"
    extract_dataset(index, dataset_dir, images_dir, feats_filename)


if __name__ == '__main__':
    main()
