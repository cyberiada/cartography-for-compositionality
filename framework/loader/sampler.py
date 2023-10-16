import torch
import torch.utils.data
from .. import utils
import numpy as np
from typing import Dict, Any
import pickle5 as pickle


def read_pickle(file_path: str) -> set:
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


class Curriculum:
    def __init__(self, starting_percent=0.04, increase_scale=1.9, total_steps=50000, batch_size=128, verbose=True):
        self.starting_percent = starting_percent
        self.increase_scale = increase_scale
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.verbose = verbose

        self.step_interval = self._set_step_interval()
        self.current_percent = starting_percent
        self.seen_instance_count = 0
        self.seen_step_count = 0
        print(self)

    def _set_step_interval(self):
        step_count = 1
        starting_percent = self.starting_percent
        while starting_percent < 1:
            starting_percent *= self.increase_scale
            step_count += 1

        step_interval = self.total_steps / step_count
        step_interval = int(step_interval)
        return step_interval

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def reset(self):
        print("Reset", self)
        self.seen_instance_count = 0
        self.seen_step_count = 0
        print("Reset", self)
        

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
            self.current_percent *= self.increase_scale
            print(self, flush=True)
            

        return change_interval_flag


class InfiniteSampler(torch.utils.data.Sampler):
    # Now, we ensured that every sample gets the same run count
    def __init__(self, data_source: torch.utils.data.Dataset, replacement=False, seed=None, indices_path=None, curriculum=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.seed = utils.seed.get_randstate(seed)
        self.indices_path = indices_path
        self.curriculum = curriculum

        # need to give this indices_path in arguments
        if self.indices_path is not None:
            self.indices = self._read_indices(indices_path)
            self.current_indices = set(self.indices[:int(len(self.indices) * self.curriculum.current_percent)])
            assert isinstance(self.indices, list)
            print(f"self.indices: {len(self.indices)}", flush=True)

        print(f"Seed is set to {self.seed}", flush=True)

    def _read_indices(self, indices_path: str) -> set:
        return read_pickle(indices_path)

    def __iter__(self):
        n = len(self.data_source)
        assert not self.replacement, f"Work in an epoch manner, self.replacement: {self.replacement}"

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
                    while sample not in self.current_indices and pos < n:
                        sample = i_list[pos]
                        pos += 1

                    if pos >= n:
                        i_list = self.seed.permutation(n).tolist()
                        pos = 0
                        sample = i_list[pos]
                        pos += 1
                        while sample not in self.current_indices:
                            sample = i_list[pos]
                            pos += 1

                    assert sample in self.current_indices, f"Sample is not in indices: {sample}"
                        
                if self.curriculum is not None:
                    change_interval_flag = self.curriculum.tick()
                    if change_interval_flag:
                        self.current_indices = set(self.indices[:int(len(self.indices) * self.curriculum.current_percent)])
                 
                yield sample

    def __len__(self):
        return 0x7FFFFFFF
