class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, nid, image_pos, text, tokens=None, filtered_tokens=None, entities=None, idx=None,
                 image_hash=None, faces_embeddings=None, feats=None, boxes=None, normal_boxes=None, attr_ids=None,
                 obj_ids=None, is_matched=None, headline=None, snippet=None, caption=None):
        self.nid = nid
        self.image_pos = image_pos
        self.text = text
        self.headline = headline
        self.snippet = snippet
        self.caption = caption
        self.tokens = tokens
        self.filtered_tokens = filtered_tokens
        self.entities = entities
        self.idx = idx
        self.image_hash = image_hash
        self.faces_embeddings = faces_embeddings
        self.feats = feats
        self.boxes = boxes
        self.normal_boxes = normal_boxes
        self.attr_ids = attr_ids
        self.obj_ids = obj_ids
        self.is_matched = is_matched  # whether the visual and obj matched
