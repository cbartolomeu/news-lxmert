import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import spacy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.lxmert_original import LxmertForPreTraining, LxmertTokenizer
from src.lxmert_news import LxmertNewsForPreTraining
from src.entities.extract import get_entities, get_ent_token_filter

from src.lxmert_optimizer import BertAdam
from src.utils.input_example import InputExample
from src.utils.input_features import InputFeatures
from src.utils.random_masks import random_word, random_feat
from src.utils.logger import log_result
from src.lxmert_metrics import LxmertMetrics


class LXMERT:
    def __init__(self, train_dset: Dataset, eval_dset: Dataset, test_dset: Dataset, train_metrics_dset: Dataset,
                 eval_metrics_dset: Dataset, test_metrics_dset: Dataset, output: str, max_seq_length: int,
                 max_faces: int,
                 batch_size: int, masked_lm_ratio: float, ent_mask_ratio: float, masked_feats_ratio: float,
                 load: str = None, load_lxmert: str = None, multi_gpu: bool = False, from_scratch: bool = False,
                 use_entities: bool = False, use_faces: bool = False):
        self.train_dset = train_dset
        self.eval_dset = eval_dset
        self.test_dset = test_dset

        self.nlp = spacy.load("en_core_web_lg")

        self.output = output

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.masked_lm_ratio = masked_lm_ratio
        self.ent_mask_ratio = ent_mask_ratio
        self.masked_feats_ratio = masked_feats_ratio
        self.use_entities = use_entities
        self.use_faces = use_faces
        self.max_faces = max_faces

        self.multi_gpu = multi_gpu

        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        self.output = f"{output}/{now}"
        os.makedirs(self.output)
        print(f"Models will be saved in {self.output}")

        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        if self.use_faces:
            self.model = LxmertNewsForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased",
                                                                  visual_feat_faces_dim=512).to(torch.device("cuda"))
        else:
            self.model = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased").to(torch.device("cuda"))

        self.train_metrics = LxmertMetrics(self.model, self.tokenizer, train_metrics_dset, self.batch_size,
                                           self.max_seq_length, use_faces, max_faces, 500)
        self.eval_metrics = LxmertMetrics(self.model, self.tokenizer, eval_metrics_dset, self.batch_size,
                                          self.max_seq_length, use_faces, max_faces, 500)
        self.test_metrics = LxmertMetrics(self.model, self.tokenizer, test_metrics_dset, self.batch_size,
                                          self.max_seq_length, use_faces, max_faces, 1000)

        # Weight initialization and loading
        if from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)
        if load is not None:
            self.__load(load)
        if load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.__load_lxmert(load_lxmert)

        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

    def __convert_example_to_features(self, example: InputExample, random_feat_fn, random_faces_fn):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        """
        tokens = self.tokenizer(example.text.strip())["input_ids"]
        ent_filter_tokens = np.zeros(len(tokens), int)

        if self.use_entities:
            marked_text_seq, entities = get_entities(self.nlp, example.text.strip())
            ent_filter_tokens = get_ent_token_filter(self.tokenizer, entities, tokens, marked_text_seq)

            # assert example.tokens is not None
            # tokens = example.tokens
            # ent_filter_tokens = example.filtered_tokens

        # Remove [CLS] and [SEP] tokens
        tokens = self.tokenizer.convert_ids_to_tokens(tokens[1:-1])
        ent_filter_tokens = ent_filter_tokens[1:-1]

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]
            ent_filter_tokens = ent_filter_tokens[:(self.max_seq_length - 2)]

        # Get random words
        masked_tokens, masked_label = random_word(tokens, ent_filter_tokens, self.tokenizer, self.masked_lm_ratio,
                                                  self.ent_mask_ratio)

        # concatenate lm labels and account for CLS, SEP, SEP
        masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)

        # Mask & Segment Word
        lm_label_ids = ([-100] + masked_label + [-100])
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-100)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(lm_label_ids) == self.max_seq_length

        # Mask Image Features:
        masked_feat, feat_mask = random_feat(example.feats, self.masked_feats_ratio, random_feat_fn)
        visual_attention_mask = np.ones(len(example.feats))

        # Face embeddings
        faces = np.zeros((self.max_faces, 512))
        masked_faces = np.zeros((self.max_faces, 512))
        faces_mask = np.zeros(self.max_faces)
        face_attention_mask = np.zeros(self.max_faces)
        if self.use_faces:
            assert example.faces_embeddings is not None
            n_faces = len(example.faces_embeddings)
            if n_faces > 0:
                max_faces = self.max_faces if self.max_faces < n_faces else n_faces
                faces_embeddings = np.array(example.faces_embeddings[0:max_faces])
                faces[0:max_faces] = faces_embeddings
                masked_faces_inner, faces_mask_inner = random_feat(faces_embeddings,
                                                                   self.masked_feats_ratio, random_faces_fn)
                masked_faces[0:max_faces] = masked_faces_inner
                faces_mask[0:max_faces] = faces_mask_inner
                face_attention_mask[0:max_faces] = np.ones(max_faces)

        return InputFeatures(
            input_ids=input_ids,
            face_feats=faces,
            face_attention_mask=face_attention_mask,
            visual_feats=masked_feat,
            visual_pos=example.boxes,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=lm_label_ids,
            visual_attention_mask=visual_attention_mask,
            obj_labels={
                "obj": (example.obj_ids, feat_mask),
                "attr": (example.attr_ids, feat_mask),
                "feat": (example.feats, feat_mask),
                "faces_feat": (faces, faces_mask),
            },
            matched_label=example.is_matched,
        )

    def __forward(self, examples, dataset):
        train_features = [self.__convert_example_to_features(example, dataset.get_random_feat, dataset.get_random_faces)
                          for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_pos for f in train_features])).cuda()
        visual_attention_masks = torch.tensor([f.visual_attention_mask for f in train_features],
                                              dtype=torch.float).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        obj_labels = {}
        for key in ('obj', 'attr', 'feat'):
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)

        faces_labels = torch.from_numpy(np.stack([f.obj_labels["faces_feat"][0] for f in train_features])).type(
            torch.FloatTensor).cuda()
        faces_mask = torch.from_numpy(np.stack([f.obj_labels["faces_feat"][1] for f in train_features])).cuda()
        obj_labels["faces_feat"] = (faces_labels, faces_mask)

        # Joint Prediction
        matched_labels = torch.tensor([f.matched_label for f in train_features], dtype=torch.long).cuda()

        faces_feats = torch.from_numpy(np.stack([f.face_feats for f in train_features])).type(torch.FloatTensor).cuda()
        faces_attention_mask = torch.tensor([f.face_attention_mask for f in train_features], dtype=torch.float).cuda()

        """
        forward(self, input_ids, visual_feats, visual_pos, attention_mask,
                 visual_attention_mask, token_type_ids, labels, obj_labels,
                 matched_label, ans):
        """
        if self.use_faces:
            output = self.model(
                input_ids=input_ids,
                visual_feats=feats,
                visual_pos=pos,
                faces_feats=faces_feats,
                faces_attention_mask=faces_attention_mask,
                visual_attention_mask=visual_attention_masks,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=lm_labels,
                obj_labels=obj_labels,
                matched_label=matched_labels,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            output = self.model(
                input_ids=input_ids,
                visual_feats=feats,
                visual_pos=pos,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=lm_labels,
                obj_labels=obj_labels,
                matched_label=matched_labels,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        return output["loss"], output["losses"]

    def __train_batch(self, batch, dataset):
        loss, losses = self.__forward(batch, dataset)
        if self.multi_gpu:
            loss = loss.mean()
            for key in losses:
                losses[key] = losses[key].mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

        for key in losses:
            losses[key] = losses[key].cpu().item()

        return loss.item(), losses

    def __valid_batch(self, batch, dataset):
        with torch.no_grad():
            loss, losses = self.__forward(batch, dataset)
            if self.multi_gpu:
                loss = loss.mean()
                for key in losses:
                    losses[key] = losses[key].mean()

            for key in losses:
                losses[key] = losses[key].cpu().item()
        return loss.item(), losses

    def __evaluate_epoch(self, eval_ds, eval_ld, iters: int = -1):
        self.model.eval()
        total_loss = 0.
        total_losses = {}
        for i, batch in enumerate(eval_ld):
            loss, losses = self.__valid_batch(batch, eval_ds)
            total_loss += loss

            for key in losses:
                total_losses[key] = losses[key] if key not in total_losses else total_losses[key] + losses[key]

            if i == iters:
                break

        avg_eval_loss = total_loss / len(eval_ld)
        print("The valid loss is %0.4f" % avg_eval_loss)

        for key in total_losses:
            total_losses[key] = total_losses[key] / len(eval_ld)

        return avg_eval_loss, total_losses

    def __save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s_LXRT.pth" % name))

    def __load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def __load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load(path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)

    def train(self, epochs: int, lr, warmup_ratio: float, accum_iter: int = 1):
        train_ld = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: x,
                              num_workers=1, pin_memory=True, drop_last=True)
        eval_ld = DataLoader(self.eval_dset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: x,
                             num_workers=1, pin_memory=True, drop_last=False)

        # Optimizer
        batch_per_epoch = len(train_ld) / accum_iter
        t_total = batch_per_epoch * epochs
        warmup_iters = int(t_total * warmup_ratio)
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print("Warm up Iters: %d" % warmup_iters)
        optimizer = BertAdam(self.model.parameters(), lr=lr, warmup=warmup_ratio, t_total=t_total)

        # Train
        best_eval_loss = 9595.

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.
            acc_loss = 0.
            total_losses = {}
            acc_losses = {}
            for i, batch in tqdm(enumerate(train_ld), total=len(train_ld)):
                loss, losses = self.__train_batch(batch, self.train_dset)

                # Update accumulate batches loss for gradient accumulation
                acc_loss += loss
                for key in losses:
                    acc_losses[key] = losses[key] if key not in acc_losses else acc_losses[key] + losses[key]

                # Update total loss
                if ((i + 1) % accum_iter == 0) or (i + 1 == batch_per_epoch):
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += acc_loss / accum_iter
                    for key in acc_losses:
                        acc_losses[key] = acc_losses[key] / accum_iter
                        total_losses[key] = acc_losses[key] if key not in total_losses else total_losses[key] + \
                                                                                            acc_losses[key]

                    # Reset accumulate batches loss
                    acc_loss = 0.
                    acc_losses = {}
            avg_train_loss = total_loss / batch_per_epoch

            for key in total_losses:
                total_losses[key] = total_losses[key] / batch_per_epoch
            print("The training loss for Epoch %d is %0.4f" % (epoch, avg_train_loss))

            avg_eval_loss, eval_losses = self.__evaluate_epoch(self.eval_dset, eval_ld, iters=-1)
            train_metrics = self.train_metrics.calculate_metrics()
            eval_metrics = self.eval_metrics.calculate_metrics()

            log_result(epoch, avg_train_loss, avg_eval_loss, train_metrics, eval_metrics)

            # Save
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                self.__save("BEST_EVAL_LOSS")
            self.__save("Epoch%02d" % (epoch + 1))

    def test(self):
        eval_metrics = self.eval_metrics.export_metrics(self.output, "eval")
        print(eval_metrics)
        test_metrics = self.test_metrics.export_metrics(self.output, "test")
        print(test_metrics)
