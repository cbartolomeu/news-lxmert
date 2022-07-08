import random
import numpy as np
import torch
import math
import csv

from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm

from src.utils.input_features import InputFeatures


def seed_worker(worker_id):
    worker_seed = 100
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class LxmertMetrics:
    def __init__(self, model, tokenizer, dset, batch_size, max_seq_length, use_faces, max_faces, n_items):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_faces = max_faces
        self.n_batches = math.ceil(n_items / self.batch_size)
        self.use_faces = use_faces

        g = torch.Generator()
        g.manual_seed(0)

        self.ld = DataLoader(dset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: x,
                             num_workers=1, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)

        self.batches = self.__get_batches()

    def __convert_example_to_features(self, example, caption=None, im_feats=None, im_boxes=None, f_embeddings=None):
        caption = example.text if caption is None else caption

        input_ids = self.tokenizer(caption.strip())["input_ids"]

        # Account for [CLS] and [SEP] with "- 2"
        if len(input_ids) > self.max_seq_length - 2:
            input_ids = input_ids[:(self.max_seq_length - 2)]

        # Mask & Segment Word
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        visual_attention_mask = np.ones(len(example.feats))

        # Face embeddings
        faces = np.zeros((self.max_faces, 512))
        face_attention_mask = np.zeros(self.max_faces)
        if self.use_faces:
            assert example.faces_embeddings is not None
            f_feats = example.faces_embeddings if f_embeddings is None else f_embeddings
            n_faces = len(f_feats)
            if n_faces > 0:
                max_faces = self.max_faces if self.max_faces < n_faces else n_faces
                faces_embeddings = np.array(f_feats[0:max_faces])
                faces[0:max_faces] = faces_embeddings
                face_attention_mask[0:max_faces] = np.ones(max_faces)

        return InputFeatures(
            input_ids=input_ids,
            visual_feats=example.feats if im_feats is None else im_feats,
            visual_pos=example.boxes if im_boxes is None else im_boxes,
            visual_attention_mask=visual_attention_mask,
            face_feats=faces,
            face_attention_mask=face_attention_mask,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )

    def __forward(self, features):
        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats for f in features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_pos for f in features])).cuda()
        visual_attention_masks = torch.tensor([f.visual_attention_mask for f in features],
                                              dtype=torch.float).cuda()

        faces_feats = torch.from_numpy(np.stack([f.face_feats for f in features])).type(torch.FloatTensor).cuda()
        faces_attention_mask = torch.tensor([f.face_attention_mask for f in features], dtype=torch.float).cuda()
        """
        forward(self, input_ids, visual_feats, visual_pos, attention_mask,
                 visual_attention_mask, token_type_ids, labels, obj_labels,
                 matched_label, ans):
        """
        output = self.model(
            input_ids=input_ids,
            visual_feats=feats,
            visual_pos=pos,
            visual_attention_mask=visual_attention_masks,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            faces_feats=faces_feats,
            faces_attention_mask=faces_attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        pred = softmax(output["cross_relationship_score"], dim=1)
        return pred[:, 1].detach().cpu().numpy()

    def __get_batches(self):
        batches = []
        for i, batch in enumerate(self.ld):
            batches.append(batch)
            if len(batches) == self.n_batches:
                break

        return batches

    def __write_csv(self, filename, headers, ids, ranks, r_scores):
        with open(filename, 'wt') as file:
            tsv_writer = csv.writer(file, delimiter="\t")

            tsv_writer.writerow(headers)
            for i in range(len(ranks)):
                rank = ranks[i]
                scores = r_scores[i]
                row = np.zeros(len(ranks) + 2)
                row[0] = ids[i]
                row[1] = rank
                row[2:] = scores[:]
                tsv_writer.writerow(row)

    def __i2t(self, batches):
        n_items = self.batch_size * (self.n_batches - 1) + len(batches[-1])
        ranks = np.zeros(n_items)
        ranks_scores = np.zeros((n_items, n_items))
        ids = np.zeros(n_items)

        for i in tqdm(range(len(batches))):
            for j in range(len(batches[i])):
                # Image to query
                query = batches[i][j]
                query_idx = i * self.batch_size + j

                scores = np.zeros(n_items)
                for batch_idx in range(len(batches)):
                    features = [
                        self.__convert_example_to_features(example=item, im_feats=query.feats, im_boxes=query.boxes,
                                                           f_embeddings=query.faces_embeddings)
                        for item in batches[batch_idx]]
                    batch_scores = self.__forward(features)

                    start_idx = self.batch_size * batch_idx
                    scores[start_idx:start_idx + len(batches[batch_idx])] = batch_scores

                ss = np.argsort(scores)[::-1]
                ranks_scores[query_idx] = ss
                ids[query_idx] = query.idx
                ranks[query_idx] = np.where(ss == query_idx)[0][0]

        print("Image -> Text")

        r1 = 100 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100 * len(np.where(ranks < 10)[0]) / len(ranks)
        r20 = 100 * len(np.where(ranks < 20)[0]) / len(ranks)
        r50 = 100 * len(np.where(ranks < 50)[0]) / len(ranks)
        r100 = 100 * len(np.where(ranks < 100)[0]) / len(ranks)

        median_r = np.floor(np.median(ranks)) + 1
        mean_r = ranks.mean() + 1
        mrr100 = np.sum(1 / (ranks[np.where(ranks < 100)[0]] + 1)) / len(ranks)

        return {
                   "r1": r1,
                   "r5": r5,
                   "r10": r10,
                   "r20": r20,
                   "r50": r50,
                   "r100": r100,
                   "median": median_r,
                   "mean": mean_r,
                   "mrr100": mrr100,
               }, ids, ranks, ranks_scores

    def __t2i(self, batches):
        n_items = self.batch_size * (self.n_batches - 1) + len(batches[-1])
        ranks = np.zeros(n_items)
        ranks_scores = np.zeros((n_items, n_items))
        ids = np.zeros(n_items)

        for i in tqdm(range(len(batches))):
            for j in range(len(batches[i])):
                # Text to query
                query = batches[i][j]
                query_idx = i * self.batch_size + j

                scores = np.zeros(n_items)
                for batch_idx in range(len(batches)):
                    features = [
                        self.__convert_example_to_features(example=item, caption=query.text)
                        for item in batches[batch_idx]]
                    batch_scores = self.__forward(features)

                    start_idx = self.batch_size * batch_idx
                    scores[start_idx:start_idx + len(batches[batch_idx])] = batch_scores

                ss = np.argsort(scores)[::-1]
                ranks_scores[query_idx] = ss
                ids[query_idx] = query.idx
                ranks[query_idx] = np.where(ss == query_idx)[0][0]

        print("Text -> Image")

        r1 = 100 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100 * len(np.where(ranks < 10)[0]) / len(ranks)
        r20 = 100 * len(np.where(ranks < 20)[0]) / len(ranks)
        r50 = 100 * len(np.where(ranks < 50)[0]) / len(ranks)
        r100 = 100 * len(np.where(ranks < 100)[0]) / len(ranks)

        median_r = np.floor(np.median(ranks)) + 1
        mean_r = ranks.mean() + 1
        mrr100 = np.sum(1 / (ranks[np.where(ranks < 100)[0]] + 1)) / len(ranks)

        return {
                   "r1": r1,
                   "r5": r5,
                   "r10": r10,
                   "r20": r20,
                   "r50": r50,
                   "r100": r100,
                   "median": median_r,
                   "mean": mean_r,
                   "mrr100": mrr100,
               }, ids, ranks, ranks_scores

    def calculate_metrics(self):
        # Image -> Text (Image Annotation)
        i2t_metrics, _, _, _ = self.__i2t(self.batches)

        # Text->Images(Image Search)
        t2i_metrics, _, _, _ = self.__t2i(self.batches)

        return {
            "i2t": i2t_metrics,
            "t2i": t2i_metrics,
        }

    def export_metrics(self, output, name):
        # Image -> Text (Image Annotation)
        i2t_metrics, i2t_ids, i2t_ranks, i2t_r_scores = self.__i2t(self.batches)

        headers = [f"r{i}" for i in range(len(i2t_ranks))]
        headers = ["idx", "rank"] + headers

        self.__write_csv(f"{output}/{name}-i2t.tsv", headers, i2t_ids, i2t_ranks, i2t_r_scores)

        # Text->Images(Image Search)
        t2i_metrics, t2i_ids, t2i_ranks, t2i_r_scores = self.__t2i(self.batches)

        self.__write_csv(f"{output}/{name}-t2i.tsv", headers, t2i_ids, t2i_ranks, t2i_r_scores)

        return {
            "i2t": i2t_metrics,
            "t2i": t2i_metrics,
        }
