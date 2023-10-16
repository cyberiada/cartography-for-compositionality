import torch
import torch.nn
import torch.optim
import framework
import torch.utils.data
import torch.cuda.amp
from typing import Optional, Dict, Any, Tuple, List, Iterable
import optimizer
from interfaces import Result, ModelInterface, EncoderDecoderResult
from tqdm import tqdm
import pickle
import os
from dataclasses import dataclass
import math
import random
import numpy as np
from framework.metrics.evaluate import evaluate_bleu


@dataclass
class LastBestMarker:
    iter: int
    loss: float
    accuracy: float

class Task:
    MAX_LENGHT_PER_BATCH = None
    valid_loaders: framework.data_structures.DotDict
    model_interface: ModelInterface
    batch_dim: int
    TRAIN_NUM_WORKERS = 2
    VALID_NUM_WORKERS = 2
    train_set: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    model: torch.nn.Module

    def create_datasets(self):
        raise NotImplementedError()

    def create_model_interface(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    @property
    def amp_enabled(self):
        return torch.cuda.is_available() and self.helper.args.amp

    @property
    def time_dim(self) -> int:
        return 1 - self.batch_dim

    def __init__(self, helper: framework.helpers.TrainingHelper):
        self.helper = helper
        self.helper.state.best_losses = {}
        self.helper.state.best_accuracies = {}
        self.valid_sets = framework.data_structures.DotDict()
        self.loss_average = framework.utils.Average()
        self.forward_time_meter = framework.utils.ElapsedTimeMeter()
        self.load_time_meter = framework.utils.ElapsedTimeMeter()
        self.plot_time_meter = framework.utils.ElapsedTimeMeter()
        self.task_name = self.init_task_name()

        if self.helper.args.lr_sched.type == "step":
            self.lr_scheduler = optimizer.StepLrSched(self.helper.args.lr, self.helper.args.lr_sched.steps,
                                                      self.helper.args.lr_sched.gamma)

        elif self.helper.args.lr_sched.type == "noam":
            self.lr_scheduler = optimizer.NoamLRSched(self.helper.args.lr, self.helper.args.state_size,
                                                      self.helper.args.lr_warmup)
        else:
            assert False

        self.indices_path, self.subset_training = self.get_indices_path()

        print(f"self.subset_training: {self.subset_training}")

        self.avg_num_chunks = framework.utils.Average()

        self.create_datasets()
        self.create_loaders()
        self.model = self.create_model()
        self.model = self.model.to(self.helper.device)
        self.create_model_interface()
        self.create_optimizer()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.helper.saver["scaler"] = self.scaler

        print(f"Total number of model parameters: {sum(p.numel() for p in self.model.parameters())}")

        self.helper.saver["model"] = self.model
        self.helper.restore()

    def init_task_name(self) -> str:
        dataset_names = ["scan", "cogs", "cfq", "dm_math", "pcfg"]
        
        split_name = self.helper.args.scan.train_split
        while not isinstance(split_name, str):
            split_name = split_name[0]

        print(f"Init task name full:", split_name)
        print(f"Init task name:", split_name)
        for name in dataset_names:
            if "scan" in self.helper.args.task:
                return name + "_" + self.helper.args.scan.train_split[0][0]
            if name in self.helper.args.task:
                return name

        assert False, "Wrong dataset name or task definition"

    def subset_training(self) -> bool:
        return self.helper.args.indices_path is not None

    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(vset, batch_size=self.test_batch_size,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS)

    def replace_valid_set(self, name: str, vset: torch.utils.data.Dataset):
        self.valid_sets[name] = vset
        self.valid_loaders[name] = self.create_valid_loader(vset)

    def create_train_loader(self, loader: torch.utils.data.Dataset, seed: Optional[int] = None) \
                            -> torch.utils.data.DataLoader:
        
        if self.indices_path is not None:
            # create training curriculum
            curriculum = framework.loader.sampler.Curriculum(
                dataset_size=len(loader),
                batch_size=self.helper.args.batch_size,
                total_steps=self.helper.args.stop_after,
                bin_count=self.helper.args.bin_count
            )
        else:
            curriculum = None

        return torch.utils.data.DataLoader(loader, batch_size=self.helper.args.batch_size,
                                           sampler=framework.loader.sampler.InfiniteSampler(
                                               loader, self.helper.args.batch_size, seed=seed, 
                                               indices_path=self.indices_path, curriculum=curriculum),
                                           collate_fn=framework.loader.collate.VarLengthCollate(
                                               batch_dim=self.batch_dim),
                                           num_workers=self.TRAIN_NUM_WORKERS, pin_memory=True)

    def set_optimizer_lr(self, lr: float):
        framework.utils.set_lr(self.optimizer, lr)
        if self.helper.state.iter % 100 == 0:
            self.helper.summary.log({"lr": lr})

    def set_linear_warmup(self, curr_step: int, n_steps: int, final: float) -> float:
        if curr_step >= n_steps:
            lr = final
        else:
            lr = final / n_steps * (curr_step+1)

        self.set_optimizer_lr(lr)
        return lr

    def get_indices_path(self) -> Tuple[str, bool]:
        if self.helper.args.indices_path is not None:
            cwd = os.getcwd()
            print(f"cwd: {cwd}, indices_path: {self.helper.args.indices_path}", flush=True)
            indices_path = os.path.join(cwd, self.helper.args.indices_path)
            print(f"new indices_path: {indices_path}", flush=True)
            return indices_path, True
        else:
            return None, False


    def set_lr(self):
        if self.helper.args.lr_sched.type == "step":
            self.set_linear_warmup(self.helper.state.iter, self.helper.args.lr_warmup,
                                   self.lr_scheduler.get(self.helper.state.iter))
        elif self.helper.args.lr_sched.type == "noam":
            self.set_optimizer_lr(self.lr_scheduler.get(self.helper.state.iter))
        else:
            assert False

    def prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.helper.to_device(data)

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0

            test = set.start_test()
            for d in tqdm(loader, "validation"):
                d = self.prepare_data(d)
                res = self.model_interface(d)
                digits = self.model_interface.decode_outputs(res)
                loss_sum += res.loss.sum().item() * res.batch_size

                test.step(digits, d)

        self.model.train()
        return test, loss_sum / len(set)

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        return self.validate_on(self.valid_sets[name], self.valid_loaders[name])

    def update_best_accuracies(self, name: str, accuracy: float, loss: float):
        if name not in self.helper.state.best_losses or loss < self.helper.state.best_losses[name].loss:
                self.helper.state.best_losses[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        if name not in self.helper.state.best_accuracies or accuracy > \
                self.helper.state.best_accuracies[name].accuracy:
            self.helper.state.best_accuracies[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        return {
            f"{name}/time_since_best_loss": self.helper.state.iter - self.helper.state.best_losses[name].iter,
            f"{name}/time_since_best_accuracy": self.helper.state.iter - self.helper.state.best_accuracies[name].iter
        }

    def validate_on_names(self, name_it: Iterable[str]) -> Dict[str, Any]:
        charts = {}
        sum_accuracy = 0
        sum_all_losses = 0

        for name in name_it:
            test, loss = self.validate_on_name(name)

            print(f"Validation accuracy on {name}: {test.accuracy}")
            charts[f"{name}/loss"] = loss
            sum_all_losses += loss
            charts.update({f"{name}/{k}": v for k, v in test.plot().items()})
            sum_accuracy += test.accuracy

            charts.update(self.update_best_accuracies(name, test.accuracy, loss))

        charts["mean_accuracy"] = sum_accuracy / len(self.valid_sets)
        charts["mean_loss"] = sum_all_losses / len(self.valid_sets)
        return charts

    def validate(self) -> Dict[str, Any]:
        return self.validate_on_names(self.valid_sets.keys())

    def plot(self, res: Result) -> Dict[str, Any]:
        plots = {}

        self.loss_average.add(res.loss)

        if self.helper.state.iter % 200 == 0:
            plots.update(res.plot())

        if self.helper.state.iter % 20 == 0:
            plots["train/loss"] = self.loss_average.get()
            plots["timing/ms_per_iter"] = self.forward_time_meter.get(True) * 1000 / 20
            plots["timing/ms_per_load"] = self.load_time_meter.get(True) * 1000 / 20
            plots["timing/ms_per_plot"] = self.plot_time_meter.get(True) * 1000 / 20

        if self.helper.state.iter % self.helper.args.test_interval == 0:
            plots.update({f"validation/{k}": v for k, v in self.validate().items()})

        if self.helper.state.iter % 20 == 0:
            plots["average_num_chunks"] = self.avg_num_chunks.get()

        return plots

    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def create_optimizer(self):
        if self.helper.args.optimizer == "adam":
            self.set_optimizer(torch.optim.Adam(self.model.parameters(), self.helper.args.lr,
                                                weight_decay=self.helper.args.wd, betas=self.helper.args.adam.betas))
        elif self.helper.args.optimizer == "sgd":
            self.set_optimizer(torch.optim.SGD(self.model.parameters(), self.helper.args.lr,
                                               weight_decay=self.helper.args.wd, momentum=0.9))
        else:
            assert False, f"Unsupported optimizer: {self.helper.args.optimizer}"

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.helper.saver.register("optimizer", self.optimizer, replace=True)

    def get_train_batch(self) -> Dict[str, Any]:
        return next(self.data_iter)

    def run_model(self, data: torch.Tensor, greedy=False) -> Tuple[Result, Dict[str, Any]]:
        teacher_forcing = not greedy
        res = self.model_interface(data, teacher_forcing=teacher_forcing)
        return res, {}

    def chunk_batch_dim(self, data: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        if n == 1:
            return [data]

        res = [{} for _ in range(n)]
        for k, v in data.items():
            assert torch.is_tensor(v), "Only tensors are supported by autosplitting"

            bd = self.batch_dim if self.batch_dim < v.ndimension() else 0
            assert v.shape[bd] % n == 0

            for i, c in enumerate(v.chunk(n, dim=bd)):
                res[i][k] = c

        return res

    def is_seq2seq_task(self, data: Dict[str, Any]) -> bool:
        return "in_len" in data and "out_len" in data

    def get_seq_length(self, data: Dict[str, Any]) -> int:
        # This assumes separate encoder and decoder
        return max(data["in"].shape[self.time_dim], data["out"].shape[self.time_dim])

    def get_n_chunks(self, data: Dict[str, Any]) -> int:
        max_length_per_batch = self.helper.args.max_length_per_batch or self.MAX_LENGHT_PER_BATCH
        if self.is_seq2seq_task(data) and max_length_per_batch:
            # The formula below assumes quadratic memory consumption
            return int(2**int(self.get_seq_length(data) / max_length_per_batch))
        return 1

    def train_step(self) -> Tuple[Result, Dict[str, Any]]:
        plots = {}

        with self.forward_time_meter:
            self.set_lr()
            data = self.prepare_data(self.get_train_batch())
            
            self.optimizer.zero_grad(set_to_none=True)

            n_chunks = self.get_n_chunks(data)
            res_list = []
            res_greedy_list = []
            weights = []

            # print(f'data["in"].shape, data["out"].shape: {data["in"].shape, data["out"].shape}')

            if "cogs" in self.helper.args.task:
                in_str = [self.train_set.in_vocabulary(s) for s in data["in"].transpose(0, 1).tolist()]
                out_str = [self.train_set.out_vocabulary(s) for s in data["out"].transpose(0, 1).tolist()]
            else:
                in_str = [self.train_set._cache.in_vocabulary(s) for s in data["in"].transpose(0, 1).tolist()]
                out_str = [self.train_set._cache.out_vocabulary(s) for s in data["out"].transpose(0, 1).tolist()]
                

            in_str: List[str] = [" ".join(s[:int(slen)]) for s, slen in zip(in_str, data["in_len"].tolist())]
            out_str: List[str] = [" ".join(s[:int(slen)]) for s, slen in zip(out_str, data["out_len"].tolist())]

            self.avg_num_chunks.add(n_chunks)

            total_out_len = data["out_len"].sum()
            for d in self.chunk_batch_dim(data, n_chunks):
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    res, custom_plots = self.run_model(d)
                    res_list.append(res)
                    plots.update(custom_plots)

                    with torch.no_grad():
                        self.model.eval()
                        res_greedy, _ = self.run_model(d, greedy=True)
                        self.model.train()

                    res_greedy_list.append(res_greedy)
                    
                weights.append((d["out_len"].sum() / total_out_len) if "out_len" in d else 1)
                assert torch.isfinite(res_list[-1].loss)
                self.scaler.scale(res_list[-1].loss * weights[-1]).backward()

            self.scaler.unscale_(self.optimizer)
            if self.helper.args.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.helper.state.iter += 1
            res = res_list[0].__class__.merge(res_list, weights)
            res_greedy = res_greedy_list[0].__class__.merge(res_greedy_list, weights)

        return res, res_greedy, plots, in_str, out_str


    def save_scores(self, res: EncoderDecoderResult, bleus: List[float], step_idx: int, out_str: List[str], epoch: int, store_path: str="scores"):
        file_types = ["chia", "ppl", "idx", "bleu"]

        store_dir_path = os.path.join(store_path, self.task_name)
        os.makedirs(store_dir_path, exist_ok=True)
        
        for ftype in file_types:
            if ftype == "chia": f_save = res.chia 
            elif ftype == "ppl": f_save = res.ppl
            elif ftype == "idx": f_save = res.idx
            elif ftype == "bleu": f_save = bleus
            else: assert False, "Unknown file type"

            file_name = os.path.join(store_dir_path, f"epoch{epoch}_stepidx{step_idx}_{ftype}.pickle")

            with open(file_name, 'wb') as handle:
                pickle.dump(f_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def update_data_indices(self, idx_to_sentences: Dict[int, Dict[str, str]], res: EncoderDecoderResult, \
                            in_str: List[str], out_str: List[str]) -> Dict[int, Dict[str, str]]:

        idx = res.idx.tolist()
        new_idx_to_sentences = {i: {'in': s_in, 'out': s_out} for i, s_in, s_out in zip(idx, in_str, out_str)}
        idx_to_sentences.update(new_idx_to_sentences)
        return idx_to_sentences


    def save_idx_to_sentences(self, idx_to_sentences: Dict[int, Dict[str, str]], store_path: str="scores", add_arg=""):
        store_dir_path = os.path.join(store_path, self.task_name)
        os.makedirs(store_dir_path, exist_ok=True)
        file_name = os.path.join(store_dir_path, f"idx_to_sentences{add_arg}.pickle")

        with open(file_name, 'wb') as handle:
            pickle.dump(idx_to_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def report_bleu(self, res: EncoderDecoderResult, out_str: List[str], verbose=False) -> List[float]:
        pred_seq = torch.argmax(res.outputs, -1)

        if "cogs" in self.helper.args.task:
            pred_seq = [self.train_set.out_vocabulary(s) for s in pred_seq.transpose(0, 1).tolist()]
        else:
            pred_seq = [self.train_set._cache.out_vocabulary(s) for s in pred_seq.transpose(0, 1).tolist()]

        pred_str: List[str] = [" ".join(s[:int(slen)]) for s, slen in zip(pred_seq, res.out_lengths.tolist())]

        bleus = evaluate_bleu(pred_str, out_str)

        if verbose:
            print("\n\n==========================================================================\n")
            for pred_s, out_s, bleu in zip(pred_str, out_str, bleus):
                print("Prediction: ", pred_s)
                print("Target: ", out_s)
                print("BLEU: ", bleu)
                print()
            print("\n==========================================================================\n\n")
            print(f"bleus: {np.mean(bleus)}")

        return bleus


    def train(self):
        self.loss_average.reset()

        epoch = 0
        epoch_loss, step_count = 0, 0
        batch_count = math.ceil(len(self.train_set) / self.helper.args.batch_size)

        self.data_iter = iter(self.train_loader)
        
        self.train_loader.sampler.curriculum.reset()

        idx_to_sentences: Dict[int, Dict[str, str]] = {} # idx -> {"in": "Who is ..?", "out": "SELECT DISTINCT .."}

        pbar = tqdm(range(self.helper.args.stop_after or 10e10))
        for step_idx in pbar:
            if step_idx % batch_count == 0:
                if step_idx != 0:
                    self.helper.summary.log({"train_loss_epoch": epoch_loss / step_count}, step=epoch)
                epoch_loss, step_count = 0, 0
                epoch += 1
                pbar.set_description(f"Training: Epoch {epoch}")

            self.load_time_meter.stop()

            res, res_greedy, plots, in_str, out_str = self.train_step()

            plots.update(self.plot(res))

            if not self.subset_training:
                verbose = (step_idx % batch_count) == 0 and (step_idx // batch_count) < 10 
                bleus = self.report_bleu(res_greedy, out_str, verbose=verbose)
                self.save_scores(res, bleus, step_idx, out_str, epoch)

            epoch_loss += res.loss
            step_count += 1

            if epoch <= 2 and not self.subset_training:
                self.update_data_indices(idx_to_sentences, res, in_str, out_str)

            with self.plot_time_meter:
                self.helper.summary.log(plots)

            self.load_time_meter.start()

            self.helper.tick()

        if not self.subset_training:
            self.save_idx_to_sentences(idx_to_sentences)

    @property
    def test_batch_size(self):
        return self.helper.args.test_batch_size or self.helper.args.batch_size
