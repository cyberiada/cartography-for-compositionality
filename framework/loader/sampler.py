import torch
import torch.utils.data
from .. import utils
import numpy as np
from typing import Dict, Any
import pickle5 as pickle
from itertools import chain
from random import shuffle
from math import ceil


def read_pickle(file_path: str) -> set:
    with open(file_path, "rb") as handle:
        return pickle.load(handle)

class Label:
    DIFFICULTY = "diffs"
    IN_LEN = "inlens" 
    OUT_LEN = "outlens"

class Curriculum:
    def __init__(self, dataset_size, batch_size, total_steps, bin_count=20, full_when=0.5, verbose=True):
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.bin_count = bin_count
        self.verbose = verbose
        self.dataset_size = dataset_size
        self.full_when = full_when
        self.step_interval = int((total_steps * full_when) // bin_count)
        self.dataset_increase_step = ceil(dataset_size / bin_count) 

        self.seen_instance_count = 0
        self.seen_step_count = 0
        self.phase = 1

        self.avail_dataset_size = self._set_avail_dataset_size()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def reset(self):
        print("Reset", self)
        self.seen_instance_count = 0
        self.seen_step_count = 0
        self.phase = 1
        print("Reset", self)

    def _set_avail_dataset_size(self):
        avail_size = self.dataset_increase_step * self.phase
        index_cap = self.batch_size - (avail_size % self.batch_size)
        return int(avail_size + index_cap)

    def tick(self):
        self.seen_instance_count += 1
        step_start_flag = self.seen_instance_count % self.batch_size == 0
        if step_start_flag:
            self.seen_step_count += 1
        
        interval_start_flag = (self.seen_step_count % self.step_interval == 0)
        change_interval_flag = interval_start_flag and step_start_flag
        
        if change_interval_flag:
            print(self, flush=True)
            print(f"step_start_flag, interval_start_flag, change_interval_flag: {step_start_flag, interval_start_flag, change_interval_flag}", flush=True)
            self.phase += 1
            self.avail_dataset_size = self._set_avail_dataset_size()
            print(self, flush=True)

        return change_interval_flag



class InfiniteSampler(torch.utils.data.Sampler):
    # Now, we ensured that every sample gets the same run count
    def __init__(self, data_source: torch.utils.data.Dataset, batch_size: int, replacement=False, seed=None, indices_path=None, curriculum=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.seed = utils.seed.get_randstate(seed)
        self.indices_path = indices_path
        self.batch_size = batch_size
        self.curriculum = curriculum

        if self.indices_path is not None:
            self.indices = self._read_indices()
            self.difficulties = self._read_column(Label.DIFFICULTY)
            self.inlens = self._read_column(Label.IN_LEN)
            self.outlens = self._read_column(Label.OUT_LEN)

            assert isinstance(self.indices, list)
            print(f"self.indices: {len(self.indices)}", flush=True)

        print(f"Seed is set to {self.seed}", flush=True)

    def _create_batches(self, sorted_indices):
        for i in range(0, len(sorted_indices), self.batch_size):
            yield sorted_indices[i:i+self.batch_size]

    def _shuffle_indices(self, sorted_indices):
        batches = list(self._create_batches(sorted_indices))
        shuffle(batches)
        batches = list(chain(*batches))
        return batches

    def _get_sorted_indices(self):
        avail_size = self.curriculum.avail_dataset_size

        print(f"avail_size: {avail_size}")
        batches = self.indices[:avail_size]
        batch_lens = self.inlens[:avail_size]

        sorted_indices = [index for _, index in sorted(zip(batch_lens, batches), key=lambda p: p[0])]
        sorted_indices = self._shuffle_indices(sorted_indices)
        return sorted_indices

    def _read_indices(self) -> set:
        return read_pickle(self.indices_path)

    def _read_column(self, label: str) -> str:
        dirs = self.indices_path.split("/")
        dirs[-1] = "_".join([label, dirs[-1]])
        path = "/".join(dirs)
        return read_pickle(path)
        
    def __iter__(self):
        n = self.curriculum.avail_dataset_size
        assert not self.replacement, f"Work in an epoch manner, self.replacement: {self.replacement}"

        i_list = None
        pos = n
        while True:
            if pos >= n:
                pos = 0
                i_list = self._get_sorted_indices()

            sample = i_list[pos]
            pos += 1

            if self.curriculum is not None:
                change_interval_flag = self.curriculum.tick()
                if change_interval_flag:
                    i_list = self._get_sorted_indices()
                    n = len(i_list)
                
            yield sample

    def __len__(self):
        return 0x7FFFFFFF
