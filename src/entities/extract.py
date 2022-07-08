import numpy as np

def get_entities(nlp, text_seq):
    entities = []

    doc = nlp(text_seq)

    modified_text_seq = ""
    last_ent_end_char = 0
    for ent in doc.ents:
        ent_info = {
            'start': ent.start_char,
            'end': ent.end_char,
            'text': ent.text,
            'label': ent.label_,
        }

        entities.append(ent_info)

        modified_text_seq = f"{modified_text_seq}{text_seq[last_ent_end_char:ent.start_char]}[MASK] {text_seq[ent.start_char: ent.end_char]} [MASK]"

        last_ent_end_char = ent.end_char

    return modified_text_seq, entities


def get_ent_token_filter(tokenizer, entities, orig_tokens, marked_text_seq):
    marked_tokens = tokenizer(marked_text_seq.strip())["input_ids"]

    entities_tokens_mask = np.zeros(len(orig_tokens), dtype=int)

    orig_tokens_idx, ent_idx = 0, 0
    ent_token_start = 0
    found = False
    for token_idx in range(len(marked_tokens)):
        token = marked_tokens[token_idx]
        # Token is [MASK]
        if token == 103 and not found:
            found = True
            ent_token_start = orig_tokens_idx
        elif token == 103 and found:
            found = False
            entities[ent_idx]["token_start"] = ent_token_start
            entities[ent_idx]["token_end"] = orig_tokens_idx - 1
            ent_idx += 1
        else:
            if found:
                entities_tokens_mask[orig_tokens_idx] = 1
            orig_tokens_idx += 1

    return entities_tokens_mask
