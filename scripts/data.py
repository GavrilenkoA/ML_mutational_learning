# %%
import os
import pandas as pd
import numpy as np
import re
# %%



# %%
def get_desc(df):
    phys_embed = list(df.repr)
    ref_pattern = re.compile(r'\[ [\S\n\t\v ]*?\]')
    all_features = []
    for item in phys_embed:
        ref_content = ref_pattern.findall(item)
        descr_seq = []
        for feature in ref_content:
            arr = np.array([float(ch) for ch in re.sub(r'\[|\]|\n', "", feature).split()])
            descr_seq.append(arr.reshape(1, -1))
        descr_seq = np.concatenate(descr_seq, axis = 0)
        all_features.append(descr_seq)
    for item in all_features:
        assert item.shape == (24, 40)
    return all_features

# %%
