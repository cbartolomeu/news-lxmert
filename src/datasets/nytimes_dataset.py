import base64
import json
import pickle
import random

import numpy as np
from torch.utils.data import Dataset

from src.utils.input_example import InputExample


def get_text(mode, news_id, news_piece, caption):
    # Get headline
    headline = None
    if "headline" in news_piece and "main":
        if "main" in news_piece["headline"]:
            headline = news_piece["headline"]["main"]

    # Get snippet
    snippet = None
    if "snippet" in news_piece:
        snippet = news_piece["snippet"]

    # Mode 1: Text = caption
    if mode == 1:
        return caption

    # Mode 2: Text = caption + headline
    if mode == 2:
        if headline is not None:
            return f"{caption} {headline}"
        else:
            print(f"[WARN] Newspiece with id {news_id} does not have a headline field")

    # Mode 3: Text = caption + snippet
    if mode == 3:
        if snippet is not None:
            return f"{caption} {snippet}"
        else:
            print(f"[WARN] Newspiece with id {news_id} does not have an snippet field")

    # Mode 4: Text = caption + snippet + headline
    if mode == 4:
        if snippet is not None and headline is not None:
            return f"{caption} {snippet} {headline}."
        else:
            print(f"[WARN] Newspiece with id {news_id} does not have an snippet or headline field")

    # Mode 5: Text = headline
    if mode == 5:
        if headline is not None:
            return f"{headline}"
        else:
            print(f"[WARN] Newspiece with id {news_id} does not have a headline field")

    # Mode 6: Text = snippet
    if mode == 6:
        if snippet is not None:
            return f"{snippet}"
        else:
            print(f"[WARN] Newspiece with id {news_id} does not have an snippet field")

    # Mode 7: Text = snippet + headline
    if mode == 7:
        if snippet is not None and headline is not None:
            return f"{snippet} {headline}."
        else:
            print(f"[WARN] Newspiece with id {news_id} does not have an snippet or headline field")

    return ""


def decode_base64_string(string, dtype, shape):
    return np.frombuffer(base64.decodebytes(string.encode()), dtype=dtype).reshape(shape)


class NYTimesDataset(Dataset):
    def __init__(self, ds_dir: str, f_dir: str, data_index: str, mode: int):
        self.ds_dir = ds_dir
        self.f_dir = f_dir
        self.index = pickle.load(open(data_index, "rb"))
        self.mode = mode

    def get_random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.index[random.randint(0, self.__len__() - 1)]
        _, image_info = self.__get_data(datum)
        n_regions = len(image_info["roi_features"])
        return image_info["roi_features"][random.randint(0, n_regions - 1)]

    def get_random_faces(self):
        n_faces = 0
        while n_faces == 0:
            datum = self.index[random.randint(0, self.__len__() - 1)]
            _, image_info = self.__get_data(datum)
            n_faces = len(image_info["faces_embeddings"])

        if n_faces > 1:
            return image_info["faces_embeddings"][random.randint(0, n_faces - 1)]
        else:
            return image_info["faces_embeddings"][0]

    def __get_data(self, metadata, output_image=True):
        news_id, image_pos = metadata
        filename = f"{self.ds_dir}/{news_id}"

        # Read news piece
        news_piece = None
        with open(filename) as reader:
            news_piece = json.load(reader)

        assert (news_piece is not None)

        # Access image positions in sections
        image_positions = news_piece["image_positions"]
        assert (len(image_positions) > 0)

        image_position = image_positions[image_pos]

        # Get array of sections (paragraphs, captions, etc)
        parsed_sections = news_piece["parsed_section"]
        assert (len(parsed_sections) > image_position)

        # Obtain image section
        image_section = parsed_sections[image_position]
        assert (image_section["type"] == "caption")

        faces_embeddings = []
        if "facenet_details" in image_section:
            faces_embeddings = image_section["facenet_details"]["embeddings"]

        # Get caption text
        caption = image_section["text"]

        caption = get_text(self.mode, news_id, news_piece, caption)

        # Get tokens and filtered_tokens
        tokens, filtered_tokens, entities = None, None, None
        if "tokens" in image_section and "size" in image_section and "filter" in image_section and "entities" in image_section:
            tokens = decode_base64_string(image_section["tokens"], int, image_section["size"]).copy()
            filtered_tokens = decode_base64_string(image_section["filter"], int, image_section["size"]).copy()
            entities = image_section["entities"]

        text_info = {
            "caption": caption,
            "tokens": tokens,
            "filter": filtered_tokens,
            "entities": entities
        }

        # Build image path
        try:
            with open(f"{self.f_dir}/{image_section['hash']}.json") as reader:
                image_info = json.load(reader)
                n_regions = image_info["n_regions"]

                return text_info, {
                    "roi_features": decode_base64_string(image_info["roi_features"], np.float32,
                                                         (n_regions, -1)).copy(),
                    "normalized_boxes": decode_base64_string(image_info["normalized_boxes"], np.float32,
                                                             (n_regions, -1)).copy(),
                    "attr_ids": decode_base64_string(image_info["attr_ids"], np.int64, n_regions).copy(),
                    "obj_ids": decode_base64_string(image_info["obj_ids"], np.int64, n_regions).copy(),
                    "faces_embeddings": faces_embeddings,
                }
        except Exception as ex:
            print(ex)
            print(metadata)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item: int):
        metadata = self.index[item]

        text_info, image_info = self.__get_data(metadata)

        # Replace the sentence with an sentence corresponding to another image
        is_matched = 1
        if random.random() < 0.5:
            is_matched = 0
            other_datum_idx = random.randint(0, self.__len__() - 1)
            while other_datum_idx == item:
                other_datum_idx = random.randint(0, self.__len__() - 1)
            text_info, _ = self.__get_data(self.index[other_datum_idx], output_image=False)

        return InputExample(
            nid=metadata[0],
            image_pos=metadata[1],
            text=text_info["caption"],
            tokens=text_info["tokens"],
            filtered_tokens=text_info["filter"],
            entities=text_info["entities"],
            faces_embeddings=image_info["faces_embeddings"],
            feats=image_info["roi_features"],
            boxes=image_info["normalized_boxes"],
            attr_ids=image_info["attr_ids"],
            obj_ids=image_info["obj_ids"],
            is_matched=is_matched
        )


class SimpleNYTimesDataset(Dataset):
    def __init__(self, ds_dir: str, f_dir: str, data_index: str, mode: int):
        self.ds_dir = ds_dir
        self.f_dir = f_dir
        self.index = pickle.load(open(data_index, "rb"))
        self.mode = mode

    def __get_data(self, metadata):
        news_id, image_pos = metadata
        filename = f"{self.ds_dir}/{news_id}"

        # Read news piece
        news_piece = None
        with open(filename) as reader:
            news_piece = json.load(reader)

        assert (news_piece is not None)

        # Access image positions in sections
        image_positions = news_piece["image_positions"]
        assert (len(image_positions) > 0)

        image_position = image_positions[image_pos]

        # Get array of sections (paragraphs, captions, etc)
        parsed_sections = news_piece["parsed_section"]
        assert (len(parsed_sections) > image_position)

        # Obtain image section
        image_section = parsed_sections[image_position]
        assert (image_section["type"] == "caption")

        faces_embeddings = []
        if "facenet_details" in image_section:
            faces_embeddings = image_section["facenet_details"]["embeddings"].copy()

        # Get caption text
        caption = image_section["text"]

        text = get_text(self.mode, news_id, news_piece, caption)

        # Get tokens and filtered_tokens
        tokens, filtered_tokens, entities = None, None, None
        if "tokens" in image_section and "size" in image_section and "filter" in image_section and "entities" in image_section:
            tokens = decode_base64_string(image_section["tokens"], int, image_section["size"]).copy()
            filtered_tokens = decode_base64_string(image_section["filter"], int, image_section["size"]).copy()
            entities = image_section["entities"]

        # Get headline
        headline = ""
        if "headline" in news_piece and "main":
            if "main" in news_piece["headline"]:
                headline = news_piece["headline"]["main"]

        # Get snippet
        snippet = ""
        if "snippet" in news_piece:
            snippet = news_piece["snippet"]

        section_name = ""
        if "section_name" in news_piece:
            section_name = news_piece["section_name"]

        text_info = {
            "caption": caption,
            "headline": headline,
            "snippet": snippet,
            "text": text,
            "tokens": tokens,
            "filter": filtered_tokens,
            "entities": entities,
            "section_name": section_name
        }

        # Build image path
        with open(f"{self.f_dir}/{image_section['hash']}.json") as reader:
            image_info = json.load(reader)
            n_regions = image_info["n_regions"]

            return text_info, {
                "image_hash": image_section["hash"],
                "roi_features": decode_base64_string(image_info["roi_features"], np.float32,
                                                     (n_regions, -1)).copy(),
                # "boxes": decode_base64_string(image_info["boxes"], np.float32, (n_regions, -1)).copy(),
                "normalized_boxes": decode_base64_string(image_info["normalized_boxes"], np.float32,
                                                         (n_regions, -1)).copy(),
                "attr_ids": decode_base64_string(image_info["attr_ids"], np.int64, n_regions).copy(),
                "obj_ids": decode_base64_string(image_info["obj_ids"], np.int64, n_regions).copy(),
                "faces_embeddings": faces_embeddings,
            }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item: int):
        metadata = self.index[item]

        text_info, image_info = self.__get_data(metadata)

        return InputExample(
            nid=metadata[0],
            image_pos=metadata[1],
            text=text_info["text"],
            headline=text_info["headline"],
            snippet=text_info["snippet"],
            caption=text_info["caption"],
            tokens=text_info["section_name"],
            # tokens=text_info["tokens"],
            filtered_tokens=text_info["filter"],
            entities=text_info["entities"],
            faces_embeddings=image_info["faces_embeddings"],
            feats=image_info["roi_features"],
            boxes=image_info["normalized_boxes"],
            attr_ids=image_info["attr_ids"],
            obj_ids=image_info["obj_ids"],
            image_hash=image_info["image_hash"],
            # normal_boxes=image_info["boxes"],
        )
