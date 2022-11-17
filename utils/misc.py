from collections import defaultdict, deque
import numpy as np
from timm.utils import get_state_dict

import torch
import torch.distributed as dist
from torch._six import inf
from timm.utils.agc import adaptive_clip_grad
from typing import Union, Iterable
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, agc=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            #assert not (clip_grad and agc)
            if agc is not None and clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  
                #norm = AGC(parameters, agc)
                norm = adaptive_clip_grad(list(parameters)[::-2], clip_factor=agc) 
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                          
            elif clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            elif agc is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                #norm = AGC(parameters, agc)
                norm = adaptive_clip_grad(list(parameters)[::-2], clip_factor=agc)   #attention: skip parameters from classifier
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
