import torch
import torch.nn
import torch.nn.functional as F
from typing import Dict, Tuple
from models.encoder_decoder import add_eos
from models.transformer_enc_dec import TransformerResult
from ..model_interface import ModelInterface
import framework

from ..encoder_decoder import EncoderDecoderResult


class TransformerEncDecInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    @staticmethod
    def chia(outputs_data: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # outputs_data.shape: [max_len, bsz, |V|], ref.shape: [max_len, bsz], mask.shape: [max_len, bsz]
        outputs_data = F.softmax(outputs_data, dim=-1)
        probs = outputs_data.gather(dim=2, index=ref.type(torch.int64).unsqueeze(2)).squeeze(2)
        probs = probs * mask
        chia = probs.sum(dim=0) / mask.sum(dim=0)
        chia = chia.detach().cpu()
        return chia
        
    @staticmethod
    def perplexity(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        l = loss.sum(dim=0) / mask.sum(dim=0)
        l = l.detach().cpu()
        ppl = torch.exp(l)
        return ppl

    def loss(self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l = framework.layers.cross_entropy(outputs.data, ref, reduction='none', smoothing=self.label_smoothing)
        l = l.reshape_as(ref) * mask
        ppl = self.perplexity(l, mask)
        #print(f"ppl: {ppl}", flush=True)
        l = l.sum() / mask.sum() 
        return l, ppl

    def decode_outputs(self, outputs: EncoderDecoderResult) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(self, data: Dict[str, torch.Tensor], train_eos: bool = True, teacher_forcing: bool = True) -> EncoderDecoderResult:
        in_len = data["in_len"].long()
        out_len = data["out_len"].long()
        idx = data["idx"].cpu()

        #print(f"idx: {idx}", flush=True)

        in_with_eos = add_eos(data["in"], data["in_len"], self.model.encoder_eos)
        out_with_eos = add_eos(data["out"], data["out_len"], self.model.decoder_sos_eos)
        in_len += 1
        out_len += 1

        teacher_forcing = teacher_forcing and self.model.training # clearer

        # print(f"max_len: {out_len.max().item()}, out_with_eos.shape: {out_with_eos.shape}")

        res = self.model(in_with_eos.transpose(0, 1), in_len, out_with_eos.transpose(0, 1),
                         out_len, teacher_forcing=teacher_forcing, max_len=out_with_eos.shape[0])

        # This __call__ is for each decoding step, so with EncoderDecoderResult.merge they are merged
        res.data = res.data.transpose(0, 1)
        
        len_mask = ~self.model.generate_len_mask(out_with_eos.shape[0], out_len if train_eos else (out_len - 1)).\
                                                 transpose(0, 1)

        loss, ppl = self.loss(res, out_with_eos, len_mask)
        chia = self.chia(res.data, out_with_eos, len_mask)
        #print(f"chia.shape: {chia.shape}", flush=True)
        #print(f"chia: {chia}", flush=True)

        return EncoderDecoderResult(res.data, res.length, loss, chia, ppl, idx)
