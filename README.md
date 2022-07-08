# Understanding News Text and Images Connection with Context-enriched Multimodal Transformers

This is an implementation of NewsLXMERT described in the [paper](TODO:arxiv:link):

```bibtex
@inproceedings{newslxmert_acmmm22,
    author = {Bartolomeu, Cláudio and Nóbrega, Rui and Semedo, David},
    title = {Understanding News Text and Images Connection with Context-enriched Multimodal Transformers},
    year = {2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
    location = {Lisbon, Portugal},
    numpages = {10}
}
```

## Preparation

Download the [NYTimes800k](https://github.com/alasdairtran/transform-and-tell) and [NewsImages](https://github.com/NewsImagesDataset/NewsImagesDataset) datasets.

There are a set of modification steps on both datasets needed to run this NewsLXMERT implementation.

### NYTimes800k

All documents are extracted from MongoDB to optimize IO operations.

Each document is stored in one out of three possible directories, depending on its `split` field: train, valid or test.
A document file is named with the article's id (e.g. `1111-2222-333.json`).

The image features of all articles are extracted before-hand using a FRCNN into a JSON file and stored on a separate folder.
The image features JSON file must be named `<image_hash>.jpg` and has the following structure:

```json
{
    "roi_features": "<base64 encoded roi features>",
    "boxes": "<base64 encoded boxes>",
    "normalized_boxes": "<base64 encoded normalized boxes>",
    "obj_ids": "<base64 encoded obj ids>",
    "obj_probs": "<base64 encoded obj probs>",
    "attr_ids": "<base64 encoded attr ids>",
    "attr_probs": "<base64 encoded attr probs>",
    "n_regions": "<number of regions>"
}
```

In order to have an file index, there is a need to create a `.pickle` that stores a dictionary, where the keys are indexes of an array (0 to number_documents-1) and values are tuples. A tuple `t = (<aid>, <iid>)`, where `aid` is the article id and `iid` is the image index of that article (0 to number_of_image-1).

### NewsImages

The news title and text must be translated to english and placed for each article into two new fields `title_en` and `text_en`.

The image features of all articles are extracted before-hand using a FRCNN into a JSON file and stored on a separate folder.
The image features JSON file must be named `<image_hash>.jpg` and has the following structure:

```json
{
    "roi_features": "<base64 encoded roi features>",
    "boxes": "<base64 encoded boxes>",
    "normalized_boxes": "<base64 encoded normalized boxes>",
    "obj_ids": "<base64 encoded obj ids>",
    "obj_probs": "<base64 encoded obj probs>",
    "attr_ids": "<base64 encoded attr ids>",
    "attr_probs": "<base64 encoded attr probs>",
    "n_regions": "<number of regions>"
}
```

The faces features of all articles' images are identified using a MTCNN and its features are extracted using a FaceNet into the following new fields: 
* `faces_embeddings`: base64 encoded string of the faces embeddings extracted by FaceNet.
* `n_faces`: number of faces identified by MTCNN.
* `faces_detect_probs`: base64 encoded string of the faces detect probs.
* `faces_size`: size of each face feature.

## Unsupervised Training

This implementation supports cpu and single-gpu training, the latter is recommended, because is faster and simpler.

To do unsupervised pre-training of a NewsLXMERT model on NYTimes800k, run:

```bash
python -m src.train_nytimes --trainDsDir <train_split_dir> \
    --trainIndex <train_split_index_file> \
    --validDsDir <validation_split_dir> \
    --validIndex <validation_split_index_file> \
    --testDsDir <test_split_dir> \
    --testIndex <test_split_index_file> \
    --featsDir <image_features_dir> \
    --output <model_checkpoints_dir> \
    --epochs 20 \
    --batchSize 256 \
    --lr 1e-4 \
    --warmupRatio 0.05 \
    --mode 7 \
    --maskedLmRatio 0.15 \
    --maskedFeatsRatio 0.15 \
    --maxSeqLen 100 \
    --entities True \
    --faces True
```

To do unsupervised fine-tuning of a NewsLXMERT model on NewsImages, run:

```bash
python -m src.train_mediaeval --trainDsDir <train_split_dir> \
    --trainIndex <train_split_index_file> \
    --validDsDir <validation_split_dir> \
    --validIndex <validation_split_index_file> \
    --testDsDir <test_split_dir> \
    --testIndex <test_split_index_file> \
    --featsDir <image_features_dir> \
    --output <model_checkpoints_dir> \
    --epochs 20 \
    --batchSize 256 \
    --lr 1e-4 \
    --warmupRatio 0.05 \
    --mode 7 \
    --maskedLmRatio 0.15 \
    --maskedFeatsRatio 0.15 \
    --maxSeqLen 100 \
    --entities True \
    --faces True \
    --loadLxmert <.pth_file_checkpoint>
```

The previous commands can be run with the `--test` flag to evaluate NewsLXMERT in the validation and test split, all metrics are also logged.

Run `python -m src.train_nytimes --help` to read all available training parameters description.

The `--mode` is a training/test flag that specifies which text elements of the news articles text are used. This flag can have the following values:

| Mode | Text Elements Used | NYTimes800k | NewsImages |
| ---- | ----------- | :---------: | :--------: |
| 1 | Caption | X | |
| 2 | Caption + Headline | X | |
| 3 | Caption + Snippet | X | |
| 4 | Caption + Headline + Snippet | X | |
| 5 | Headline | X | X |
| 6 | Snippet | X | X |
| 7 | Headline + Snippet | X | X |

## Models

Our pre-trained NewsLXMERT models on NYTimes800k can be downloaded as following:

|  | epochs | task | mrr@100 (i2t) | mrr@100 (t2i) | model | md5 |
|-|-|-|-|-|-|-|
| NewsLXMERT | 20 | News piece-Image Matching | 0.1189 | 0.1044 | [download](https://drive.google.com/file/d/1442TL4IVwPpK9dzCQr_9eV-bhjKeyOFy/view?usp=sharing) | `8ebfb5953b52fa41ef04c5c7f61e07c4` |
| NewsLXMERT | 20 | Image-Caption Matching | 0.3342 | 0.3031 | [download](https://drive.google.com/file/d/19VRW8aeiOu_81HRZMl1jYwbdRp5B5q7G/view?usp=sharing) | `2c9efb49dea29578b3b117cb76540d90` |

Our finetuned NewsLXMERT models on NewsImages can be downloaded as following:

|  | epochs | pre-train task | mrr@100 (i2t) | mrr@100 (t2i) | model | md5 |
|-|-|-|-|-|-|-|
| NewsLXMERT | 20 | News piece-Image Matching | 0.1230 | 0.1247 | [download](https://drive.google.com/file/d/1CSVxFPGmDeq4CqXaLGrih1yElZHnJ2BJ/view?usp=sharing) | `cf5b73a8facce1e8538a6291b87cbb95` |
| NewsLXMERT | 20 | Image-Caption Matching | 0.1373 | 0.1294 | [download](https://drive.google.com/file/d/1idrlf5m2gYMPanw0zdMg5ZvxMLjDgYkQ/view?usp=sharing) | `148004368cd6eccd89a5be3b10f6e00f` |
