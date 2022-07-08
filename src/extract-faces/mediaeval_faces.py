import os

import numpy as np
import base64
import pandas as pd
import csv
from facenet import MTCNN, InceptionResnetV1
from PIL import Image
import torch

from tqdm import tqdm


def base64_encode(arr):
    return base64.b64encode(np.array(arr).tobytes()).decode("utf-8")


def write_row(writer, headers, row, embeddings, n_faces, detect_probs):
    n_headers = len(headers)

    writable_row = []
    for h_idx in range(n_headers):
        writable_row.append(row[h_idx])

    writable_row.append(base64_encode(embeddings))
    writable_row.append(n_faces)
    writable_row.append(base64_encode(detect_probs))
    writable_row.append(len(embeddings))

    writer.writerow(writable_row)


def extract_faces(tsv, images_dir, faces_dir, output):
    df = pd.read_csv(tsv, delimiter='\t')
    mtcnn = MTCNN(keep_all=True, device='cuda')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    with open(output, 'wt') as file:
        writer = csv.writer(file, delimiter="\t")

        writer.writerow(list(df.columns) + ["faces_embeddings", "n_faces", "faces_detect_probs", "faces_size"])

        for row_idx in tqdm(range(len(df))):
            row = df.iloc[row_idx]
            news_id = row["article"] if "article" in row else row["iid"]
            image_path = f"{images_dir}/{row['imgFile']}"

            try:
                img = Image.open(image_path)
                img = img.convert('RGB')
            except OSError:
                print(f"OSError on image: {image_path} from article {news_id}")
                continue

            face_path = os.path.join(faces_dir, row["imgFile"])

            with torch.no_grad():
                try:
                    faces, probs = mtcnn(img, save_path=face_path,
                                         return_prob=True)
                except IndexError:  # Strange index error on line 135 in utils/detect_face.py
                    print(f"IndexError on image: {image_path} from article {news_id}")
                    faces = None
                if faces is None:
                    write_row(writer, df.columns.values, row, [], 0, [])
                    continue
                embeddings, _ = resnet(faces)

            embeddings = embeddings.cpu().tolist()[:10]
            print(np.array(embeddings).shape)
            n_faces = len(faces)
            detect_probs = probs.tolist()[:10]

            print(np.array(embeddings).dtype)
            a = base64_encode(embeddings)
            b = decode_base64_string(a, np.float64, (len(embeddings), -1))
            print(np.array(b).shape)
            write_row(writer, df.columns.values, row, embeddings, n_faces, detect_probs)

            break


def decode_base64_string(string, dtype, shape):
    return np.frombuffer(base64.decodebytes(string.encode()), dtype=dtype).reshape(shape)


def a():
    df = pd.read_csv("/user/data/c.bartolomeu/media-eval/__train_mediaeval_no_missing_imgs.tsv", delimiter='\t')

    for row_idx in tqdm(range(len(df))):
        row = df.iloc[row_idx]

        if row["faces_size"] > 0:
            a = decode_base64_string(row["faces_embeddings"], np.float32, (row["faces_size"], -1))
            print(a.shape)


def main():
    tsv = "/user/data/c.bartolomeu/media-eval/MediaEvalNewsImagesBatch04images.tsv"
    images_dir = "/user/data/c.bartolomeu/media-eval/images"
    faces_dir = "/user/data/c.bartolomeu/media-eval/faces"
    output = "/home/c.bartolomeu/temp-test.tsv"

    #a()
    #return
    extract_faces(tsv, images_dir, faces_dir, output)


if __name__ == "__main__":
    main()
