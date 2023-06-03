import numpy as np
import re
import pandas as pd
from typing import Any

SEED = 42


def get_desc(df: pd.DataFrame) -> list[np.ndarray]:
    '''
    Function for converting string to float of entries in repr columns
    '''
    phys_embed = list(df.repr)
    ref_pattern = re.compile(r'\[ [\S\n\t\v ]*?\]')
    all_features = []
    for item in phys_embed:
        ref_content = ref_pattern.findall(item)
        descr_seq = []
        for feature in ref_content:
            arr = np.array([float(ch) for ch in re.sub(r'\[|\]|\n', "",
                                                       feature).split()])
            descr_seq.append(arr.reshape(1, -1))
        descr_seq = np.concatenate(descr_seq, axis=0)
        all_features.append(descr_seq)
    for item in all_features:
        assert item.shape == (24, 40)
    return all_features


def get_data(train: pd.DataFrame, test: pd.DataFrame, samples: int = 200, target_ab: str = None) -> \
        tuple[Any, Any, Any, Any]:
    target_df = train.loc[train['Antibody'] == target_ab]
    rest_df = train.loc[train['Antibody'] != target_ab]
    target_test = test.loc[test['Antibody'] == target_ab]
    sample_target = target_df.sample(n=samples, random_state=SEED)
    train_target = sample_target.iloc[:samples // 2, :]
    valid_target = sample_target.iloc[samples // 2:, :]
    return train_target, valid_target, target_test, rest_df
