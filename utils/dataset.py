import numpy as np
from torch.utils.data import Dataset

# encode target molecule
map_target = {'LY16': 0, 'REGN33': 1, 'REGN87': 2, 'LY555': 3, 'ACE2': 4}


class Onehot(Dataset):
    def __init__(self, df):
        self.df = df

    @staticmethod
    def _encode_seq(sequence):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in sequence]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return np.array(onehot_encoded)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        onehot_features = self._encode_seq(self.df.junction_aa.iloc[idx])
        label = self.df.Label.iloc[idx]
        return onehot_features, label


class OnehotandAB(Onehot):
    ab_map = map_target

    @classmethod
    def _code_ab(cls, ab):
        ab_idx = cls.ab_map[ab]
        return ab_idx

    def __getitem__(self, idx):
        onehot_features = self._encode_seq(self.df.junction_aa.iloc[idx])
        ab = self._code_ab(self.df.Antibody.iloc[idx])
        label = self.df.Label.iloc[idx]
        return [onehot_features, ab, label]


class Phys(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        phys_features = self.df.repr.iloc[idx]
        label = self.df.Label.iloc[idx]
        return phys_features, label


class Abencode1(Dataset):
    ab_map = map_target

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    @classmethod
    def _code_ab(cls, ab):
        pos = cls.ab_map[ab]
        ab_feature = np.zeros((24, 1))
        ab_feature[pos] = 1
        return ab_feature

    def __getitem__(self, idx):
        phys_features = self.df.repr.iloc[idx]
        ab = self.df.Antibody.iloc[idx]
        ab_feature = self._code_ab(ab)
        features = np.concatenate((phys_features, ab_feature), axis=1)
        label = self.df.Label.iloc[idx]
        return features, label


class Abencode2(Dataset):
    ab_map = map_target

    def __init__(self, df):
        self.df = df

    @classmethod
    def _code_ab(cls, ab):
        ab_idx = cls.ab_map[ab]
        return ab_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.df.repr.iloc[idx]
        ab = self._code_ab(self.df.Antibody.iloc[idx])
        label = self.df.Label.iloc[idx]
        return [features, ab, label]
