import torch
import torch.utils.data
from .. import utils
import numpy as np
from typing import Dict, Any
import pickle5 as pickle

def read_pickle(file_path: str) -> set:
    with open(file_path, "rb") as handle:
        return pickle.load(handle)

class InfiniteSampler(torch.utils.data.Sampler):
    # Now, we ensured that every sample gets the same run count
    def __init__(self, data_source: torch.utils.data.Dataset, replacement=False, seed=None, indices_path=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.seed = utils.seed.get_randstate(seed)
        self.indices_path = indices_path

        # need to give this indices_path in arguments
        if self.indices_path is not None:
            self.indices = self._read_indices(indices_path)
            print(f"self.indices: {len(self.indices)}")
        
        print("")

        print(f"Seed is set to {self.seed}", flush=True)

    def _read_indices(self, indices_path: str) -> set:
        return read_pickle(indices_path)

    def __iter__(self):
        n = len(self.data_source)
        
        if self.replacement:
            while True:
                yield self.seed.randint(0, n, dtype=np.int64)
        else:
            i_list = None
            pos = n
            while True:
                if pos >= n:
                    i_list = self.seed.permutation(n).tolist()
                    pos = 0

                sample = i_list[pos]
                pos += 1

                if self.indices_path is not None:
                    while sample not in self.indices and pos < n:
                        sample = i_list[pos]
                        pos += 1

                    if pos >= n:
                        i_list = self.seed.permutation(n).tolist()
                        pos = 0
                        sample = i_list[pos]
                        pos += 1
                        while sample not in self.indices:
                            sample = i_list[pos]
                            pos += 1

                    assert sample in self.indices, f"Sample is not in indices: {sample}"
                    
                yield sample

    def __len__(self):
        return 0x7FFFFFFF
