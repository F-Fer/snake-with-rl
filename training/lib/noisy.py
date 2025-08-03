import torch
import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np

class NoisyLinear(nn.Module):
    """
    Factorised NoisyLinear layer - Fortunato et al. (2017)
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        # μ and σ parameters
        self.weight_mu   = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma= nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_eps', torch.empty(out_features, in_features))
        if bias:
            self.bias_mu    = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_eps', torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_eps', None)

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range,  mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt_()   # f(x)=sign(x)*√|x|

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        if self.bias_eps is not None:
            self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = None if self.bias_mu is None else \
                self.bias_mu  + self.bias_sigma  * self.bias_eps
        else:                       # evaluation ⇒ deterministic net
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)