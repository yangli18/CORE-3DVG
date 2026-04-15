import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, endswith_relu=False):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            hidden_dims = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(
                nn.Linear(in_dim, out_dim)
                for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
            )
        else:
            self.layers = []
        self.endswith_relu = endswith_relu

    def forward(self, x):
        if self.endswith_relu:
            for layer in self.layers:
                x = F.relu(layer(x))
            return x

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class MLPConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 use_bn=False, transpose_dims=(-1, -2), dropout=-1):
        super().__init__()
        layers = []
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        for idx, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
            if idx != num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.transpose_dims = transpose_dims

    def forward(self, x):
        if x.dim() == 4:
            dim1, dim2, transposed_dim1, transposed_dim2 = x.shape
            out_shape = [dim1, dim2, transposed_dim1, -1]
        elif x.dim() == 3:
            dim1, transposed_dim1, transposed_dim2 = x.shape
            out_shape = [dim1, transposed_dim1, -1]
        else:
            raise NotImplementedError
        x = x.transpose(*self.transpose_dims).reshape(-1, transposed_dim2, transposed_dim1)
        x = self.layers(x)
        x = x.transpose(*self.transpose_dims).reshape(*out_shape)
        return x


class TwoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, concat_result=True):
        super().__init__()
        out_dim1, out_dim2 = output_dim
        if isinstance(num_layers, list):
            num_layers1, num_layers2 = num_layers
        else:
            num_layers1 = num_layers2 = num_layers
        self.mlp1 = MLP(input_dim, hidden_dim, out_dim1, num_layers1)
        self.mlp2 = MLP(input_dim, hidden_dim, out_dim2, num_layers2)
        self.concat_result = concat_result

    def forward(self, x):
        x1 = self.mlp1(x)
        x2 = self.mlp2(x)
        if self.concat_result:
            return torch.cat([x1, x2], dim=-1)
        return x1, x2


class TwoMLPConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 use_bn=False, transpose_dims=(-1, -2), dropout=-1):
        super().__init__()
        out_dim1, out_dim2 = output_dim
        self.mlp1 = MLPConv(input_dim, hidden_dim, out_dim1, num_layers, use_bn, transpose_dims, dropout)
        self.mlp2 = MLPConv(input_dim, hidden_dim, out_dim2, num_layers, use_bn, transpose_dims, dropout)

    def forward(self, x):
        return torch.cat([self.mlp1(x), self.mlp2(x)], dim=-1)


MLP_MODULES = {
    'MLP': MLP,
    'MLP_conv': MLPConv,
    'TwoMLP': TwoMLP,
    'TwoMLPconv': TwoMLPConv,
}


class RelFeatsMHAttention(nn.Module):
    def __init__(self, d_model, h, dropout, q1_dim, q2_dim, k_dim, v_dim, rel_dim=None,
                 attn_use_key=False, separate_q=False, v_use_rel_feats=False,
                 return_raw_attention=False):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.scaling = float((q1_dim + q2_dim) // h) ** -0.5
        self.return_raw_attention = return_raw_attention
        self.attn_use_key = attn_use_key
        self.separate_q = separate_q
        self.v_use_rel_feats = v_use_rel_feats
        self.q1_dim = q1_dim
        self.q2_dim = q2_dim

        self.q_proj_weight = Parameter(torch.Tensor(q1_dim + q2_dim, d_model))
        self.q_proj_bias = Parameter(torch.empty(q1_dim + q2_dim))
        if attn_use_key:
            self.k_proj_weight = Parameter(torch.Tensor(k_dim, d_model))
            self.k_proj_bias = Parameter(torch.empty(k_dim))
        self.v_proj_weight = Parameter(torch.Tensor(v_dim, d_model))
        self.v_proj_bias = Parameter(torch.empty(v_dim))
        if self.v_use_rel_feats:
            self.rel_v_proj_weight = Parameter(torch.Tensor(v_dim, rel_dim))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = None
        self.dropout_p = dropout
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.constant_(self.q_proj_bias, 0.0)
        nn.init.constant_(self.v_proj_bias, 0.0)
        if self.attn_use_key:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.constant_(self.k_proj_bias, 0.0)
        if self.v_use_rel_feats:
            nn.init.xavier_uniform_(self.rel_v_proj_weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, rel_feats, value, key_padding_mask=None):
        tgt_len, batch_size, _ = query.size()
        src_len, _, _ = key.size()

        q = query.matmul(self.q_proj_weight.t()) + self.q_proj_bias
        v = value.matmul(self.v_proj_weight.t()) + self.v_proj_bias
        q = q * self.scaling
        if self.separate_q:
            q1, q2 = q[..., :self.q1_dim], q[..., self.q1_dim:]
        else:
            q1 = q2 = q

        q1 = q1.reshape(tgt_len, batch_size * self.h, -1).transpose(0, 1)
        q2 = q2.reshape(tgt_len, batch_size * self.h, -1).transpose(0, 1)
        v = v.view(src_len, batch_size * self.h, -1).transpose(0, 1)
        if self.v_use_rel_feats:
            rel_v = rel_feats.matmul(self.rel_v_proj_weight.t())
            rel_v = rel_v.view(tgt_len, src_len, batch_size * self.h, -1).permute(2, 0, 1, 3)

        rel_feats = rel_feats.view(tgt_len, src_len, batch_size * self.h, -1).permute(2, 0, 3, 1)
        attn_weights = torch.matmul(q1[:, :, None], rel_feats).view(batch_size * self.h, tgt_len, src_len)

        if self.attn_use_key:
            k = key.matmul(self.k_proj_weight.t()) + self.k_proj_bias
            k = k.view(src_len, batch_size * self.h, -1).permute(1, 2, 0)
            attn_weights = attn_weights + torch.matmul(q2, k)
        else:
            attn_weights = attn_weights + key.sum() * 0

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(-1, tgt_len, src_len)

        raw_attn_weights = attn_weights
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        if self.v_use_rel_feats:
            v = v[:, None] + rel_v
            attn_output = torch.matmul(attn_weights[:, :, None], v).view(batch_size * self.h, tgt_len, -1)
        else:
            attn_output = torch.bmm(attn_weights, v)
        self.attn = attn_weights

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, -1)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        if self.return_raw_attention:
            return attn_output, raw_attn_weights
        return attn_output, attn_weights


ATTN_MODULES = {
    'RelFeatsMHAttention': RelFeatsMHAttention,
}


def build_module_from_cfg(module_dict, module_cfg):
    if isinstance(module_cfg, dict):
        cfg = module_cfg.copy()
        return module_dict[cfg.pop('type')](**cfg)
    return None
