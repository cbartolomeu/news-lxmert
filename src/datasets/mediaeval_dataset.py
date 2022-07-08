import base64
import json
import random
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

from src.utils.input_example import InputExample


def decode_base64_string(string, dtype, shape):
    return np.frombuffer(base64.decodebytes(string.encode()), dtype=dtype).reshape(shape)


DATA_SPLITS = {
    "development": "__train_mediaeval_no_missing_imgs.tsv",
    "evaluation": ("_prediction_mediaeval_articles", "_MediaEvalNewsImagesBatch04images_no_missing_imgs.tsv")
}


class MediaEvalDataset(Dataset):
    def __init__(self, ds_dir: str, f_dir: str, data_index: str,
                 val_split: float = 500, test_split: float = 1000):
        self.ds_dir = ds_dir
        self.f_dir = f_dir
        self.data_index = data_index
        self.val_split = val_split
        self.test_split = test_split

        self.__prepare_splits()

    def __prepare_splits(self):

        if self.ds_dir in ["train", "valid", "test"]:
            split_file = DATA_SPLITS["development"]
            self.df = pd.read_csv(os.path.join(self.data_index, split_file), delimiter='\t')

            samples = np.arange(len(self.df))
            X_devel, X_test = train_test_split(samples, test_size=self.test_split, random_state=42)
            X_train, X_val, = train_test_split(X_devel, test_size=self.val_split, random_state=42)

            if self.ds_dir == "train":
                self.index = self.df.iloc[X_train]
            elif self.ds_dir == "valid":
                self.index = self.df.iloc[X_val]
            elif self.ds_dir == "test":
                self.index = self.df.iloc[X_test]
            else:
                raise Exception("Invalid data split.")
        else:
            split_file = DATA_SPLITS["evaluation"]
            self.index_articles = pd.read_csv(os.path.join(self.data_index, split_file[0]), delimiter='\t')
            self.index_images = pd.read_csv(os.path.join(self.data_index, split_file[1]), delimiter='\t')

    def get_random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.index.iloc[random.randint(0, self.__len__() - 1)]
        _, image_info = self.__get_data(datum)
        n_regions = len(image_info["roi_features"])
        return image_info["roi_features"][random.randint(0, n_regions - 1)]

    def get_random_faces(self):
        n_faces = 0
        while n_faces == 0:
            datum = self.index.iloc[random.randint(0, self.__len__() - 1)]
            _, image_info = self.__get_data(datum)
            n_faces = len(image_info["faces_embeddings"])

        if n_faces > 1:
            return image_info["faces_embeddings"][random.randint(0, n_faces - 1)]
        else:
            return image_info["faces_embeddings"][0]

    def __get_data(self, metadata, output_image=True):

        # Get caption text
        caption = metadata["title_en"] + " " + metadata["text_en"]

        # Get tokens and filtered_tokens
        tokens, filtered_tokens, entities = None, None, None
        if "tokens" in metadata and "size" in metadata and "filter" in metadata and "entities" in metadata:
            tokens = decode_base64_string(metadata["tokens"], int, metadata["size"]).copy()
            filtered_tokens = decode_base64_string(metadata["filter"], int, metadata["size"]).copy()
            entities = metadata["entities"]

        text_info = {
            "caption": caption,
            "tokens": tokens,
            "filter": filtered_tokens,
            "entities": entities
        }

        # Get image hash
        image_hash = metadata["imgFile"].replace(".jpg", "")

        if not output_image:
            return text_info, None

        face_embeddings = None
        if "faces_size" in metadata:
            faces_size = metadata["faces_size"]
            if faces_size > 0:
                face_embeddings = decode_base64_string(metadata["faces_embeddings"], np.float64,
                                                       (faces_size, -1)).copy()
            else:
                face_embeddings = []

        # Build image path
        try:
            with open(f"{self.f_dir}/{image_hash}.json") as reader:
                image_info = json.load(reader)
                n_regions = image_info["n_regions"]

                return text_info, {
                    "roi_features": decode_base64_string(image_info["roi_features"], np.float32,
                                                         (n_regions, -1)).copy(),
                    "normalized_boxes": decode_base64_string(image_info["normalized_boxes"], np.float32,
                                                             (n_regions, -1)).copy(),
                    "attr_ids": decode_base64_string(image_info["attr_ids"], np.int64, n_regions).copy(),
                    "obj_ids": decode_base64_string(image_info["obj_ids"], np.int64, n_regions).copy(),
                    "faces_embeddings": face_embeddings,
                }
        except Exception as ex:
            print(ex)
            print(metadata)

    def __len__(self):
        if self.ds_dir in ["train", "valid", "test"]:
            return len(self.index)
        else:
            return len(self.index_articles)

    def __getitem__(self, item: int):
        metadata = self.index.iloc[item]

        text_info, image_info = self.__get_data(metadata)

        # Replace the sentence with an sentence corresponding to another image
        is_matched = 1
        if random.random() < 0.5:
            is_matched = 0
            other_datum_idx = random.randint(0, self.__len__() - 1)
            while other_datum_idx == item:
                other_datum_idx = random.randint(0, self.__len__() - 1)
            text_info, _ = self.__get_data(self.index.iloc[other_datum_idx], output_image=False)

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


class SimpleMediaEvalDataset(Dataset):
    def __init__(self, ds_dir: str, f_dir: str, data_index: str,
                 val_split: float = 500, test_split: float = 1000):
        self.ds_dir = ds_dir
        self.f_dir = f_dir
        self.data_index = data_index
        self.val_split = val_split
        self.test_split = test_split

        self.__prepare_splits()

    def __prepare_splits(self):

        if self.ds_dir in ["train", "valid", "test"]:
            split_file = DATA_SPLITS["development"]
            self.df = pd.read_csv(os.path.join(self.data_index, split_file), delimiter='\t')

            samples = np.arange(len(self.df))
            X_devel, X_test = train_test_split(samples, test_size=self.test_split, random_state=42)
            X_train, X_val, = train_test_split(X_devel, test_size=self.val_split, random_state=42)

            if self.ds_dir == "train":
                self.index = self.df.iloc[X_train]
            elif self.ds_dir == "valid":
                self.index = self.df.iloc[X_val]
            elif self.ds_dir == "test":
                self.index = self.df.iloc[X_test]
            else:
                raise Exception("Invalid data split.")
        else:
            split_file = DATA_SPLITS["evaluation"]
            self.index_articles = pd.read_csv(os.path.join(self.data_index, split_file[0]), delimiter='\t')
            self.index_images = pd.read_csv(os.path.join(self.data_index, split_file[1]), delimiter='\t')

    def __get_data(self, metadata):

        # Get caption text
        caption = metadata["title_en"] + " " + metadata["text_en"]

        # Get tokens and filtered_tokens
        tokens, filtered_tokens = None, None
        if "tokens" in metadata and "size" in metadata and "filter" in metadata:
            tokens = decode_base64_string(metadata["tokens"], int, metadata["size"]).copy()
            filtered_tokens = decode_base64_string(metadata["filter"], int, metadata["size"]).copy()

        text_info = {
            "caption": caption,
            "tokens": tokens,
            "filter": filtered_tokens,
        }

        face_embeddings = None
        if "faces_size" in metadata:
            faces_size = metadata["faces_size"]
            if faces_size > 0:
                face_embeddings = decode_base64_string(metadata["faces_embeddings"], np.float64,
                                                       (faces_size, -1)).copy()
            else:
                face_embeddings = []

        # Get image hash
        image_hash = metadata["imgFile"].replace(".jpg", "")

        # Build image path
        with open(f"{self.f_dir}/{image_hash}.json") as reader:
            image_info = json.load(reader)
            n_regions = image_info["n_regions"]

            return text_info, {
                "roi_features": decode_base64_string(image_info["roi_features"], np.float32,
                                                     (n_regions, -1)).copy(),
                "boxes": decode_base64_string(image_info["boxes"], np.float32, (n_regions, -1)).copy(),
                "normalized_boxes": decode_base64_string(image_info["normalized_boxes"], np.float32,
                                                         (n_regions, -1)).copy(),
                "attr_ids": decode_base64_string(image_info["attr_ids"], np.int64, n_regions).copy(),
                "obj_ids": decode_base64_string(image_info["obj_ids"], np.int64, n_regions).copy(),
                "faces_embeddings": face_embeddings,
            }, image_hash

    def __len__(self):
        if self.ds_dir in ["train", "valid", "test"]:
            return len(self.index)
        else:
            return len(self.index_articles)

    def __getitem__(self, item: int):
        metadata = self.index.iloc[item]

        text_info, image_info, image_hash = self.__get_data(metadata)

        return InputExample(
            idx=item,
            image_hash=image_hash,
            nid=metadata[0],
            image_pos=metadata[1],
            text=text_info["caption"],
            tokens=text_info["tokens"],
            filtered_tokens=text_info["filter"],
            faces_embeddings=image_info["faces_embeddings"],
            feats=image_info["roi_features"],
            boxes=image_info["normalized_boxes"],
            attr_ids=image_info["attr_ids"],
            obj_ids=image_info["obj_ids"],
            normal_boxes=image_info["boxes"],
        )
