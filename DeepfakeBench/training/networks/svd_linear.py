'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection
'''


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None, gate_init_bias=3.0):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.gate_init_bias = gate_init_bias

        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.S_residual = None
        self.U_residual = None
        self.V_residual = None
        self.S_r = None
        self.U_r = None
        self.V_r = None
        self.gate_param = None

        self.weight_original_fnorm = None
        self.weight_main_fnorm = None

    def forward(self, x):
        if (self.S_residual is not None) and (self.U_residual is not None) and (self.V_residual is not None):
            gates = torch.sigmoid(self.gate_param)
            S_gated = gates * self.S_residual
            residual_weight = self.U_residual @ torch.diag(S_gated) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self):
        if (self.S_residual is not None) and (self.U_r is not None):
            device = self.weight_main.device
            gates = torch.sigmoid(self.gate_param) if (self.gate_param is not None) else None
            if gates is not None:
                sqrt_g = torch.sqrt(gates + 1e-8)
                U_res_scaled = self.U_residual * sqrt_g.unsqueeze(0)
                V_res_scaled = self.V_residual * sqrt_g.unsqueeze(0)
            else:
                U_res_scaled = self.U_residual
                V_res_scaled = self.V_residual

            if self.U_r is not None:
                U_concat = torch.cat((self.U_r, U_res_scaled), dim=1)
            else:
                U_concat = U_res_scaled

            if self.V_r is not None:
                V_concat = torch.cat((self.V_r, V_res_scaled), dim=0)
            else:
                V_concat = V_res_scaled

            UUT = U_concat @ U_concat.t()
            VVT = V_concat @ V_concat.t()
            UUT_identity = torch.eye(UUT.size(0), device=device)
            VVT_identity = torch.eye(VVT.size(0), device=device)
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
            return loss
        else:
            return torch.tensor(0.0, device=self.weight_main.device)

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            if (self.gate_param is not None):
                S_gated = torch.sigmoid(self.gate_param) * self.S_residual
                weight_current = self.weight_main + self.U_residual @ torch.diag(S_gated) @ self.V_residual
            else:
                weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            return torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
        else:
            return torch.tensor(0.0, device=self.weight_main.device)


def replace_with_svd_residual(module, r, gate_init_bias=3.0):
    if not isinstance(module, nn.Linear):
        return module

    in_features = module.in_features
    out_features = module.out_features
    bias = module.bias is not None

    new_module = SVDResidualLinear(in_features, out_features, r, bias=bias,
                                   init_weight=module.weight.data.clone(),
                                   gate_init_bias=gate_init_bias)

    if bias and module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)

    new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

    W = module.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    r_keep = min(r, S.shape[0])
    U_r = U[:, :r_keep]
    S_r = S[:r_keep]
    Vh_r = Vh[:r_keep, :]

    weight_main = U_r @ torch.diag(S_r) @ Vh_r
    new_module.weight_main.data.copy_(weight_main)
    new_module.weight_main_fnorm = torch.norm(weight_main, p='fro')

    U_residual = U[:, r_keep:]
    S_residual = S[r_keep:]
    Vh_residual = Vh[r_keep:, :]

    if S_residual.numel() > 0:
        new_module.S_residual = nn.Parameter(S_residual.clone())
        new_module.U_residual = nn.Parameter(U_residual.clone())
        new_module.V_residual = nn.Parameter(Vh_residual.clone())
        new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
        new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
        new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)

        S_residual_normalized = S_residual / (S_residual.max() + 1e-8)
        gate_init = gate_init_bias + 1.0 * S_residual_normalized
        new_module.gate_param = nn.Parameter(gate_init)
    else:
        new_module.S_residual = None
        new_module.U_residual = None
        new_module.V_residual = None
        new_module.S_r = None
        new_module.U_r = None
        new_module.V_r = None
        new_module.gate_param = None

    return new_module


def apply_svd_residual_to_self_attn(model, r, gate_init_bias=3.0):
    for name, module in list(model.named_children()):
        if 'self_attn' in name:
            for sub_name, sub_module in list(module.named_modules()):
                if isinstance(sub_module, nn.Linear):
                    parent = module
                    parts = sub_name.split('.')
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], replace_with_svd_residual(sub_module, r, gate_init_bias=gate_init_bias))
        else:
            apply_svd_residual_to_self_attn(module, r, gate_init_bias=gate_init_bias)

    for pname, p in model.named_parameters():
        if any(k in pname for k in ['S_residual', 'U_residual', 'V_residual', 'gate_param']):
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model