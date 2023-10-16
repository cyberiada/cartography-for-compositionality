import torch
from dataclasses import dataclass
from .result import Result
from typing import List, Optional


@dataclass
class EncoderDecoderResult(Result):
    outputs: torch.Tensor
    out_lengths: torch.Tensor
    loss: torch.Tensor
    chia: torch.Tensor
    ppl: torch.Tensor
    idx: torch.Tensor
    
    batch_dim = 1

    @staticmethod
    def merge(l: List, batch_weights: Optional[List[float]] = None):
        if len(l) == 1:
            # print(f"len(l): {len(l)}")
            # r = l[0]
            # print(f"r.loss.shape: {r.loss.shape}", flush=True)
            # print(f"r.outputs.shape: {r.outputs.shape}", flush=True)
            # print(f"r.out_lengths.shape: {r.out_lengths.shape}", flush=True)
            # print(f"r.chia.shape: {r.chia.shape}", flush=True)
            # print(f"r.idx.shape: {r.idx.shape}", flush=True)
            # print(f"r.ppl.shape: {r.ppl.shape}", flush=True)
            return l[0]

        batch_weights = batch_weights if batch_weights is not None else [1] * len(l)
        loss = sum([r.loss * w for r, w in zip(l, batch_weights)]) / sum(batch_weights)
        out = torch.stack([r.outputs for r in l], l[0].batch_dim)
        out = out.reshape(out.shape[0], -1, out.shape[-1]) # torch.stack([r.outputs for r in l], l[0].batch_dim)
        lens = torch.cat([r.out_lengths for r in l], 0) # torch.stack([r.out_lengths for r in l], 0)
        chias = torch.cat([r.chia for r in l], 0)
        idxs = torch.cat([r.idx for r in l], 0)
        ppls = torch.cat([r.ppl for r in l], 0)

        # print(f"loss.shape: {loss.shape}", flush=True)
        # print(f"out.shape: {out.shape}", flush=True)
        # print(f"lens.shape: {lens.shape}", flush=True)

        # print(f"chias.shape: {chias.shape}", flush=True)
        # print(f"idxs.shape: {idxs.shape}", flush=True)
        # print(f"ppls.shape: {ppls.shape}", flush=True)

        # for r in l:
        #     print(f"len(l): {len(l)}")
        #     print(f"r.out_lengths.shape: {r.out_lengths.shape}", flush=True)
        #     print(f"r.chia.shape: {r.chia.shape}", flush=True)
        #     print(f"r.idx.shape: {r.idx.shape}", flush=True)
        #     print(f"r.ppl.shape: {r.ppl.shape}", flush=True)

        # print("At merge function below 1st if clause!", flush=True)
        return_var = l[0].__class__(out, lens, loss, chias, ppls, idxs)
        return return_var
