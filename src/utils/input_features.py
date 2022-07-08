class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, visual_feats, visual_pos, attention_mask, token_type_ids, face_feats, face_attention_mask,
                 visual_attention_mask=None, labels=None, obj_labels=None, matched_label=None, ans=None):
        self.input_ids = input_ids
        self.visual_feats = visual_feats
        self.visual_pos = visual_pos
        self.attention_mask = attention_mask
        self.visual_attention_mask = visual_attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.obj_labels = obj_labels
        self.matched_label = matched_label
        self.ans = ans
        self.face_feats = face_feats
        self.face_attention_mask = face_attention_mask
