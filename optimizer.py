from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer

from math import sqrt, pow


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            if group['max_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(group["params"], group['max_grad_norm'])
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                if 't' not in state:
                    state['t'] = 0
                state['t'] += 1
                t = state['t']

                if 'm' not in state or 'v' not in state:
                    state['m'] = torch.zeros_like(grad)
                    state['v'] = torch.zeros_like(grad)

                assert len(self.defaults["betas"]) == 2
                beta_one, beta_two = self.defaults["betas"]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                state['m'].lerp_(grad, 1-beta_one)
                state['v'].lerp_(grad.square(), 1-beta_two)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                alpha_t = alpha * sqrt(1-pow(beta_two, t))/(1-pow(beta_one, t))

                # Update parameters
                p.data.addcdiv_(state['m'], state['v'].sqrt()+self.defaults['eps'], value=-alpha_t)

                # Add weight decay after the main gradient-based updates.
                p.data.sub_(p.data, alpha=(alpha*self.defaults['weight_decay']))

                self.state[p] = state

        return loss
