import random

import pandas as pd
from torch.utils.data import Sampler


class BalancedDatasetSampler(Sampler):
    """
    自定义的采样器，这个采样器会自动把不均衡样本上采样到1:1
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.indices = list(range(len(dataset)))

        df = pd.DataFrame()
        df["label"] = dataset.get_labels()
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        neg_count = label_to_count[0]
        pos_count = label_to_count[1]

        pos_indices = df.index[:pos_count].tolist()
        neg_indices = df.index[pos_count:].tolist()

        new_indices = []

        for i in range(neg_count):
            new_indices.append(random.choice(pos_indices))

        new_indices += neg_indices
        random.shuffle(new_indices)
        self.new_indices = new_indices

    def __iter__(self):
        return iter(self.new_indices)

    def __len__(self):
        return len(self.dataset)
