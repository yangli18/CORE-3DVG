from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .box_ops import box_cxcyczwhd_to_xyzxyz
from .modules import MLP_MODULES, ATTN_MODULES, build_module_from_cfg


def _get_clones(module, num_layers):
    return nn.ModuleList([deepcopy(module) for _ in range(num_layers)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerDecoderWithRelFeat(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        extra_info=None,
    ):
        if extra_info is None:
            extra_info = {}
        output = tgt
        rel_feats = extra_info.get('spatio_rel_feats', None)
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                rel_feats=rel_feats,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerDecoderLayerWithRelFeat(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False, self_attn_cfg=None):
        super().__init__()
        if self_attn_cfg is None:
            self_attn_cfg = {}
        if self_attn_cfg.get('type', 'default') == 'default':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = build_module_from_cfg(ATTN_MODULES, self_attn_cfg)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        rel_feats: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        if isinstance(self.self_attn, nn.MultiheadAttention):
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, rel_feats=rel_feats, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            raise NotImplementedError
        return self.forward_post(*args, **kwargs)


class TransformerDecoderLayerForRelFeats(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False, self_attn_cfg=None):
        super().__init__()
        if self_attn_cfg is None:
            self_attn_cfg = {}
        if self_attn_cfg:
            if self_attn_cfg.get('type', 'default') == 'default':
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            else:
                self.self_attn = build_module_from_cfg(ATTN_MODULES, self_attn_cfg)
        else:
            self.self_attn = None
        self.self_attn_method = self_attn_cfg.get('self_attn_method', 'full_attn')
        if self_attn_cfg.get('self_attn2', False):
            self.self_attn2 = deepcopy(self.self_attn)
        else:
            self.self_attn2 = None

        if self.self_attn_method == 'cross-shaped_attn':
            n1, n2 = self_attn_cfg.get('N1xN2', (24, 24))
            rows = torch.arange(0, n1)
            cols = torch.arange(0, n2)
            rr, cc = torch.meshgrid(rows, cols)
            grids = torch.stack((rr, cc), dim=-1).view(-1, 2)
            mask = (grids[:, None] != grids[None]).all(dim=-1)
            self.register_buffer('self_attn_mask', mask)

        if self.self_attn_method == 'full_attn_exclude_selfpair':
            n1, n2 = self_attn_cfg.get('N1xN2', (24, 24))
            rows = torch.arange(0, n1)
            cols = torch.arange(0, n2)
            rr, cc = torch.meshgrid(rows, cols)
            grids = torch.stack((rr, cc), dim=-1).view(-1, 2)
            mask = grids[..., 0] == grids[..., 1]
            mask = mask[None].repeat(n1 * n2, 1)
            self.register_buffer('self_attn_mask', mask)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        n1, n2, batch_size, _ = tgt.shape
        if self.self_attn is not None:
            if self.self_attn_method == 'full_attn':
                tgt = tgt.view(n1 * n2, batch_size, -1)
                if query_pos is not None:
                    query_pos = query_pos.view(n1 * n2, batch_size, -1)
                q = k = self.with_pos_embed(tgt, query_pos)
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            elif self.self_attn_method == 'full_attn_exclude_selfpair':
                tgt = tgt.view(n1 * n2, batch_size, -1)
                if query_pos is not None:
                    query_pos = query_pos.view(n1 * n2, batch_size, -1)
                q = k = self.with_pos_embed(tgt, query_pos)
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=self.self_attn_mask, key_padding_mask=tgt_key_padding_mask)[0]
            elif self.self_attn_method == 'col_attn':
                tgt = tgt.view(n1, n2 * batch_size, -1)
                if query_pos is not None:
                    query_pos = query_pos.view(n1, n2 * batch_size, -1)
                q = k = self.with_pos_embed(tgt, query_pos)
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            elif self.self_attn_method == 'row_attn':
                tgt_r = tgt.transpose(0, 1).reshape(n2, n1 * batch_size, -1)
                query_pos_r = query_pos.transpose(0, 1).reshape(n2, n1 * batch_size, -1) if query_pos is not None else None
                q_r = k_r = self.with_pos_embed(tgt_r, query_pos_r)
                tgt2 = self.self_attn(q_r, k_r, value=tgt_r)[0]
                tgt2 = tgt2.view(n2, n1, batch_size, -1).transpose(0, 1)
            elif self.self_attn_method == 'parallel row/col attn':
                tgt_c = tgt.view(n1, n2 * batch_size, -1)
                tgt_r = tgt.transpose(0, 1).reshape(n2, n1 * batch_size, -1)
                query_pos_c = query_pos.view(n1, n2 * batch_size, -1) if query_pos is not None else None
                query_pos_r = query_pos.transpose(0, 1).reshape(n2, n1 * batch_size, -1) if query_pos is not None else None
                q_c = k_c = self.with_pos_embed(tgt_c, query_pos_c)
                tgt2_c = self.self_attn(q_c, k_c, value=tgt_c)[0]
                self_attn_r = self.self_attn2 if self.self_attn2 is not None else self.self_attn
                q_r = k_r = self.with_pos_embed(tgt_r, query_pos_r)
                tgt2_r = self_attn_r(q_r, k_r, value=tgt_r)[0]
                tgt2 = tgt2_c.view(n1, n2, batch_size, -1) + tgt2_r.view(n2, n1, batch_size, -1).transpose(0, 1)
            elif self.self_attn_method == 'cross-shaped_attn':
                tgt = tgt.view(n1 * n2, batch_size, -1)
                if query_pos is not None:
                    query_pos = query_pos.view(n1 * n2, batch_size, -1)
                q = k = self.with_pos_embed(tgt, query_pos)
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=self.self_attn_mask)[0]
            else:
                raise NotImplementedError
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt = tgt.view(n1 * n2, batch_size, -1)
        if query_pos is not None:
            query_pos = query_pos.view(n1 * n2, batch_size, -1)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt.view(n1, n2, batch_size, -1)

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            raise NotImplementedError
        return self.forward_post(*args, **kwargs)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, n_heads=8, dim_feedforward=256, pt_pos_mlp_args=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        if pt_pos_mlp_args:
            self.pt_pos_mlp = MLP_MODULES[pt_pos_mlp_args.pop('type')](**pt_pos_mlp_args)
        else:
            self.pt_pos_mlp = None

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.pt_pos_mlp is not None:
            pos = self.pt_pos_mlp(pos)
            pos = pos.transpose(0, 1)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src = self.norm2(src + self.dropout2(self.ffn(src)))
        return src


class DecoderWithPredHead(nn.Module):
    def __init__(self, layer, num_layers, num_query=128, query_dim=288,
                 update_query_pos='none', init_query_pos='', init_query_src='empty',
                 sample_pt_query_proj=None, separate_query_pos_mlp=False,
                 query_pos_mlp=None, detach_query_pos_boxes=False,
                 separate_pred_head=False, pred_head=None, score_head=None,
                 return_intermediate=True):
        super().__init__()
        if sample_pt_query_proj is None:
            sample_pt_query_proj = {}
        if query_pos_mlp is None:
            query_pos_mlp = {}
        if pred_head is None:
            pred_head = {}
        if score_head is None:
            score_head = {}

        self.layers = build_layers_from_cfg(layer, num_layers)
        self.num_query = num_query
        self.query_dim = query_dim
        self.update_query_pos = update_query_pos
        self.separate_query_pos_mlp = separate_query_pos_mlp
        self.query_pos_mlp = build_module_from_cfg(MLP_MODULES, query_pos_mlp)
        if separate_query_pos_mlp:
            self.query_pos_mlp = _get_clones(self.query_pos_mlp, num_layers)
        self.detach_query_pos_boxes = detach_query_pos_boxes
        self.sample_pt_query_proj = build_module_from_cfg(MLP_MODULES, sample_pt_query_proj)
        self.return_intermediate = return_intermediate

        if pred_head:
            self.separate_pred_head = separate_pred_head
            self.pred_head = build_module_from_cfg(MLP_MODULES, pred_head)
            self.score_head = build_module_from_cfg(MLP_MODULES, score_head)
            if separate_pred_head:
                self.pred_head = _get_clones(self.pred_head, num_layers)
                self.score_head = _get_clones(self.score_head, num_layers)

        self.init_query_pos = init_query_pos
        self.init_query_src = init_query_src
        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        points_feats,
        points_pos_embed,
        text_feats,
        text_padding_mask,
        prop_boxes,
        prop_padding_mask,
        sample_pt_feat,
        sample_pt_boxes,
        sample_pt_xyz,
        ext_input=None,
    ):
        del prop_boxes, prop_padding_mask
        if ext_input is None:
            ext_input = {}
        _, batch_size, _ = points_feats.shape
        sample_pt_boxes = sample_pt_boxes.transpose(0, 1)
        sample_pt_xyz = sample_pt_xyz.transpose(0, 1)

        if self.init_query_pos != 'sample_pt_boxes':
            raise ValueError('Only sample_pt_boxes init_query_pos is supported.')
        query_pos_boxes = sample_pt_boxes
        sample_pt_padding_mask = torch.zeros(batch_size, self.num_query, device=query_pos_boxes.device, dtype=torch.bool)
        query_padding_mask = sample_pt_padding_mask
        result_dict = {'init_boxes': query_pos_boxes}

        if self.detach_query_pos_boxes:
            query_pos_boxes = query_pos_boxes.detach()

        if self.init_query_src != 'sample_pt_feat':
            raise ValueError('Only sample_pt_feat init_query_src is supported.')
        tgt_query = self.sample_pt_query_proj(sample_pt_feat.transpose(0, 1))

        all_pred_logits, all_pred_boxes, all_pred_box_mask = [], [], []
        for layer_idx, layer_module in enumerate(self.layers):
            if self.separate_query_pos_mlp:
                query_pos_embed = self.query_pos_mlp[layer_idx](query_pos_boxes)
            else:
                query_pos_embed = self.query_pos_mlp(query_pos_boxes)

            tgt_query = layer_module(
                tgt_query,
                query_pos_embed,
                query_padding_mask,
                points_feats,
                points_pos_embed,
                text_feats,
                text_padding_mask,
                ext_input,
            )

            if self.separate_pred_head:
                pred_boxes = self.pred_head[layer_idx](tgt_query)
                pred_logits = self.score_head[layer_idx](tgt_query)
            else:
                pred_boxes = self.pred_head(tgt_query)
                pred_logits = self.score_head(tgt_query)

            if self.update_query_pos == 'none':
                pred_boxes = query_pos_boxes
                pred_box_mask = torch.zeros_like(query_padding_mask, dtype=torch.bool)
            elif self.update_query_pos == 'xyz + offset, size':
                pred_boxes[..., :3] += sample_pt_xyz
                pred_box_mask = torch.ones_like(sample_pt_padding_mask, dtype=torch.bool)
            else:
                raise NotImplementedError
            query_pos_boxes = pred_boxes.detach()

            if self.return_intermediate:
                all_pred_boxes.append(pred_boxes)
                all_pred_logits.append(pred_logits)
                all_pred_box_mask.append(pred_box_mask)

        if self.return_intermediate:
            result_dict.update(
                {
                    'pred_boxes': torch.stack(all_pred_boxes),
                    'pred_logits': torch.stack(all_pred_logits),
                    'pred_box_mask': torch.stack(all_pred_box_mask),
                    'tgt_queries': tgt_query,
                }
            )
            return result_dict

        result_dict.update(
            {
                'pred_boxes': pred_boxes.unsqueeze(0),
                'pred_logits': pred_logits.unsqueeze(0),
                'pred_box_mask': pred_box_mask.unsqueeze(0),
                'tgt_queries': tgt_query,
            }
        )
        return result_dict


class TransformerDecoderLayer2CrossAttn(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1,
                 use_self_attn=True, other_args=None):
        super().__init__()
        del other_args
        self.text_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_t = nn.Dropout(dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.vis_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_v = nn.Dropout(dropout)
        self.norm_v = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(288, 8, dropout=0.1)
            self.norm1 = nn.LayerNorm(288)
            self.dropout1 = nn.Dropout(0.1)
        else:
            self.self_attn = None

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_query, query_pos_embed, query_padding_mask,
                points_feats, points_pos, text_feats, text_padding_mask, extra_info):
        del extra_info
        if self.self_attn is not None:
            self_info = self.self_attn(
                tgt_query + query_pos_embed,
                tgt_query + query_pos_embed,
                tgt_query,
                attn_mask=None,
                key_padding_mask=query_padding_mask,
            )[0]
            tgt_query = self.norm1(tgt_query + self.dropout1(self_info))

        text_info = self.text_attn(
            query=self.with_pos_embed(tgt_query, query_pos_embed),
            key=text_feats,
            value=text_feats,
            attn_mask=None,
            key_padding_mask=text_padding_mask,
        )[0]
        tgt_query = self.norm_t(tgt_query + self.dropout_t(text_info))

        vis_info = self.vis_attn(
            query=tgt_query + query_pos_embed,
            key=points_feats + points_pos,
            value=points_feats + points_pos,
            attn_mask=None,
            key_padding_mask=None,
        )[0]
        tgt_query = self.norm_v(tgt_query + self.dropout_v(vis_info))
        tgt_query = self.norm(tgt_query + self.dropout(self.ffn(tgt_query)))
        return tgt_query


class DecoderBasedMatcherWithRelModel(nn.Module):
    def __init__(
        self,
        d_model,
        dec_layer_cfg,
        decoder_cfg=None,
        detach_tgt_query=False,
        score_pred_cfg=None,
        query_pos_mlp=None,
        spatio_rel_feats_cfg=None,
        ctx_feat_proj=None,
        rel_feat_fusion='concat_ctx_feats',
        rel_feat_pos='tgt_ctx_box_embed',
        tgt_pos_mlp=None,
        ctx_pos_mlp=None,
        rel_feat_matcher=None,
        rel_score_pred=None,
        use_updated_rel_feat=True,
        detach_rel_feat=False,
    ):
        super().__init__()
        if decoder_cfg is None:
            decoder_cfg = {}
        if score_pred_cfg is None:
            score_pred_cfg = {}
        if query_pos_mlp is None:
            query_pos_mlp = {}
        if spatio_rel_feats_cfg is None:
            spatio_rel_feats_cfg = {}
        if tgt_pos_mlp is None:
            tgt_pos_mlp = {}
        if ctx_pos_mlp is None:
            ctx_pos_mlp = {}
        if rel_feat_matcher is None:
            rel_feat_matcher = {}
        if rel_score_pred is None:
            rel_score_pred = {}

        matcher_type = decoder_cfg.pop('type')
        if matcher_type not in ['TransformerDecoder', 'TransformerDecoderWithRelFeat']:
            raise ValueError(f'Unsupported matcher type: {matcher_type}')
        match_layer = build_module_from_cfg(_MODULES, dec_layer_cfg)
        norm = nn.LayerNorm(d_model)
        self.matcher = _MODULES[matcher_type](decoder_layer=match_layer, norm=norm, **decoder_cfg)

        self.detach_tgt_query = detach_tgt_query
        self.score_pred = build_module_from_cfg(MLP_MODULES, score_pred_cfg)
        self.query_pos_mlp = build_module_from_cfg(MLP_MODULES, query_pos_mlp)

        self.encode_spatio_rel_feats = len(spatio_rel_feats_cfg) > 0
        if self.encode_spatio_rel_feats:
            self.args_to_build_spatio_rel_feats = spatio_rel_feats_cfg.get('args_to_build_spatio_rel_feats')
            self.spatio_rel_feats_mlp = build_module_from_cfg(MLP_MODULES, spatio_rel_feats_cfg['mlp_cfg'])
            dist_norm_expo = spatio_rel_feats_cfg.get('dist_norm_expo', 3.0)
            if spatio_rel_feats_cfg.get('dist_norm_expo_learnable', False):
                self.dist_norm_expo = nn.Parameter(data=torch.tensor(dist_norm_expo), requires_grad=True)
            else:
                self.dist_norm_expo = dist_norm_expo
            dxdydz_norm_expo = spatio_rel_feats_cfg.get('dxdydz_norm_expo', 3.0)
            if spatio_rel_feats_cfg.get('dxdydz_norm_expo_learnable', False):
                self.dxdydz_norm_expo = nn.Parameter(data=torch.tensor(dxdydz_norm_expo), requires_grad=True)
            else:
                self.dxdydz_norm_expo = dxdydz_norm_expo

        self.ctx_feat_proj = build_module_from_cfg(MLP_MODULES, ctx_feat_proj) if ctx_feat_proj is not None else None
        self.rel_feat_fusion = rel_feat_fusion
        self.rel_feat_pos = rel_feat_pos
        if self.rel_feat_pos in ['tgt_ctx_box_embed', 'tgt_ctx_pos_embed']:
            self.tgt_pos_mlp = build_module_from_cfg(MLP_MODULES, tgt_pos_mlp)
            self.ctx_pos_mlp = build_module_from_cfg(MLP_MODULES, ctx_pos_mlp)

        rel_feat_matcher_type = rel_feat_matcher.pop('type')
        if rel_feat_matcher_type == 'TransformerDecoder':
            match_layer_cfg = rel_feat_matcher.pop('layer')
            match_layer = build_module_from_cfg(_MODULES, match_layer_cfg)
            norm = nn.LayerNorm(d_model)
            self.rel_feat_matcher = TransformerDecoder(decoder_layer=match_layer, norm=norm, **rel_feat_matcher)
        elif rel_feat_matcher_type == 'none':
            self.rel_feat_matcher = None
        else:
            self.rel_feat_matcher = build_module_from_cfg(_MODULES, rel_feat_matcher)

        self.rel_score_pred = build_module_from_cfg(MLP_MODULES, rel_score_pred) if rel_score_pred else None
        self.use_updated_rel_feat = use_updated_rel_feat
        self.detach_rel_feat = detach_rel_feat
        self.forward_cache = {}
        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _build_relation_inputs(self, inst_feats, inst_pos_boxes, query_pos_boxes, ext_input):
        num_queries = query_pos_boxes.shape[0]
        tgt_query_pos = self.query_pos_mlp(query_pos_boxes)
        extra_info = {}
        all_point_xyz = ext_input.get('all_point_xyz')
        scene_min_xyz = all_point_xyz.min(dim=1)[0]
        scene_max_xyz = all_point_xyz.max(dim=1)[0]
        scene_whd = (scene_max_xyz - scene_min_xyz).clamp(min=0.1)

        query_boxes_xyzxyz = box_cxcyczwhd_to_xyzxyz(query_pos_boxes)
        inst_boxes_xyzxyz = box_cxcyczwhd_to_xyzxyz(inst_pos_boxes)
        spatio_rel_feats = build_spatio_rel_feats(
            query_boxes_xyzxyz,
            inst_boxes_xyzxyz,
            scene_min_xyz,
            scene_whd,
            dist_norm_expo=self.dist_norm_expo,
            dxdydz_norm_expo=self.dxdydz_norm_expo,
            scene_max_xyz=scene_max_xyz,
            all_point_xyz=all_point_xyz,
            **self.args_to_build_spatio_rel_feats,
        )
        spatio_rel_feats = self.spatio_rel_feats_mlp(spatio_rel_feats)
        extra_info['spatio_rel_feats'] = spatio_rel_feats

        if self.rel_feat_pos == 'tgt_ctx_box_embed':
            tgt_pos = self.tgt_pos_mlp(query_pos_boxes)
            ctx_pos = self.ctx_pos_mlp(inst_pos_boxes)
            rel_feat_pos = tgt_pos[:, None] + ctx_pos[None, :]
        elif self.rel_feat_pos == 'tgt_ctx_pos_embed':
            assert (query_pos_boxes == inst_pos_boxes).all()
            tgt_pos = self.tgt_pos_mlp(tgt_query_pos)
            ctx_pos = self.ctx_pos_mlp(tgt_query_pos)
            rel_feat_pos = tgt_pos[:, None] + ctx_pos[None, :]
        elif self.rel_feat_pos == 'None':
            rel_feat_pos = None
        else:
            raise NotImplementedError

        ctx_feats = self.ctx_feat_proj(inst_feats) if self.ctx_feat_proj is not None else inst_feats
        if self.rel_feat_fusion == 'concat_ctx_feats':
            ctx_feats = ctx_feats[None].expand(num_queries, -1, -1, -1)
            rel_feats = torch.cat((spatio_rel_feats, ctx_feats), dim=-1)
        elif self.rel_feat_fusion == 'add_ctx_feats':
            ctx_feats = ctx_feats[None].expand(num_queries, -1, -1, -1)
            rel_feats = spatio_rel_feats + ctx_feats
        elif self.rel_feat_fusion == 'add_ctx_feats & pos_feats':
            ctx_feats = ctx_feats[None].expand(num_queries, -1, -1, -1)
            rel_feats = spatio_rel_feats + ctx_feats + rel_feat_pos
        elif self.rel_feat_fusion == 'add_ctx_feats & pos_feats & unset_pos_feats':
            ctx_feats = ctx_feats[None].expand(num_queries, -1, -1, -1)
            rel_feats = spatio_rel_feats + ctx_feats + rel_feat_pos
            rel_feat_pos = None
        else:
            raise NotImplementedError

        return tgt_query_pos, rel_feats, rel_feat_pos, extra_info

    def forward(self, inst_feats, inst_pos_boxes, text_feats, text_padding_mask,
                tgt_query, query_pos_boxes, query_padding_mask, ext_input=None):
        del query_padding_mask
        if ext_input is None:
            ext_input = {}
        num_queries, _, _ = tgt_query.shape
        if self.detach_tgt_query:
            tgt_query = tgt_query.detach()

        tgt_query_pos, rel_feats, rel_feat_pos, extra_info = self._build_relation_inputs(
            inst_feats, inst_pos_boxes, query_pos_boxes, ext_input
        )
        self.forward_cache = {
            'tgt_query_pos': tgt_query_pos,
            'rel_feats': rel_feats,
            'rel_feat_pos': rel_feat_pos,
            'extra_info': extra_info,
        }

        if self.rel_feat_matcher is not None:
            rel_feats = self.rel_feat_matcher(
                tgt=rel_feats,
                memory=text_feats,
                memory_key_padding_mask=text_padding_mask,
                pos=None,
                query_pos=rel_feat_pos,
            )
        if self.rel_score_pred is not None:
            num_layers = rel_feats.shape[0]
            batch_size = tgt_query.shape[1]
            rel_match_scores = self.rel_score_pred(rel_feats).view(
                num_layers, num_queries, -1, batch_size
            ).transpose(2, 3)
        else:
            rel_match_scores = None

        if self.detach_rel_feat:
            rel_feats = rel_feats.detach()
        extra_info['spatio_rel_feats'] = rel_feats[-1] if self.use_updated_rel_feat else rel_feats

        if isinstance(self.matcher, TransformerDecoder):
            output = self.matcher(
                tgt_query,
                memory=text_feats,
                memory_key_padding_mask=text_padding_mask,
                pos=None,
                query_pos=tgt_query_pos,
            )
        else:
            output = self.matcher(
                tgt_query,
                memory=text_feats,
                memory_key_padding_mask=text_padding_mask,
                pos=None,
                query_pos=tgt_query_pos,
                extra_info=extra_info,
            )

        match_scores = self.score_pred(output).view(-1, num_queries, text_feats.shape[1], 1)
        return {'pred_logits': match_scores, 'rel_pred_logits': rel_match_scores}

    def forward_rand_text(self, inst_feats, inst_pos_boxes, rand_text_feats, rand_text_padding_mask,
                          tgt_query, query_pos_boxes, query_padding_mask, ext_input=None):
        del inst_feats, inst_pos_boxes, query_pos_boxes, query_padding_mask, ext_input
        num_queries, _, _ = tgt_query.shape
        if self.detach_tgt_query:
            tgt_query = tgt_query.detach()

        tgt_query_pos = self.forward_cache['tgt_query_pos']
        rel_feats = self.forward_cache['rel_feats']
        rel_feat_pos = self.forward_cache['rel_feat_pos']
        extra_info = self.forward_cache['extra_info']

        if self.rel_feat_matcher is not None:
            rel_feats = self.rel_feat_matcher(
                tgt=rel_feats,
                memory=rand_text_feats,
                memory_key_padding_mask=rand_text_padding_mask,
                pos=None,
                query_pos=rel_feat_pos,
            )
        if self.rel_score_pred is not None:
            num_layers = rel_feats.shape[0]
            batch_size = tgt_query.shape[1]
            rel_match_scores = self.rel_score_pred(rel_feats).view(
                num_layers, num_queries, -1, batch_size
            ).transpose(2, 3)
        else:
            rel_match_scores = None

        extra_info['spatio_rel_feats'] = rel_feats[-1] if self.use_updated_rel_feat else rel_feats
        if isinstance(self.matcher, TransformerDecoder):
            output = self.matcher(
                tgt_query,
                memory=rand_text_feats,
                memory_key_padding_mask=rand_text_padding_mask,
                pos=None,
                query_pos=tgt_query_pos,
            )
        else:
            output = self.matcher(
                tgt_query,
                memory=rand_text_feats,
                memory_key_padding_mask=rand_text_padding_mask,
                pos=None,
                query_pos=tgt_query_pos,
                extra_info=extra_info,
            )

        match_scores = self.score_pred(output).view(-1, num_queries, rand_text_feats.shape[1], 1)
        return {'pred_logits': match_scores, 'rel_pred_logits': rel_match_scores}


class Vis2TextCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, vis_proj_args, txt_proj_args,
                 norm_before_proj=True, inner_prod_w_bias=False, vis_query_with_pos=False):
        super().__init__()
        self.text_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_t = nn.Dropout(dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.vis_proj = MLP_MODULES[vis_proj_args.pop('type')](**vis_proj_args)
        self.txt_proj = MLP_MODULES[txt_proj_args.pop('type')](**txt_proj_args)
        self.norm_before_proj = norm_before_proj
        if inner_prod_w_bias:
            self.inner_prod_bias = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        else:
            self.inner_prod_bias = None
        self.vis_query_with_pos = vis_query_with_pos

    def forward(self, vis_query, vis_pos, text_feats, text_padding_mask, extra_info=None):
        del extra_info
        query = vis_query + vis_pos if self.vis_query_with_pos else vis_query
        text_info = self.text_attn(query=query, key=text_feats, value=text_feats, key_padding_mask=text_padding_mask)[0]
        text_query = self.norm_t(self.dropout_t(text_info))
        proj_vis_query = self.vis_proj(vis_query)
        proj_txt_query = self.txt_proj(text_query)
        return correlation(proj_vis_query, proj_txt_query, self.norm_before_proj, self.inner_prod_bias)


def correlation(x, y, norm_before_proj=False, bias=None):
    if norm_before_proj:
        corr_score = (F.normalize(x, p=2, dim=-1) * F.normalize(y, p=2, dim=-1)).sum(dim=-1, keepdim=True)
    else:
        corr_score = (x * y).sum(dim=-1, keepdim=True)
    if bias is not None:
        corr_score = corr_score + bias
    return corr_score


def get_box_property(boxes):
    centers = (boxes[..., :3] + boxes[..., 3:]) / 2
    sizes = (boxes[..., 3:] - boxes[..., :3]).clamp(min=1e-6)
    volumes = sizes.prod(dim=-1, keepdim=True).clamp(min=1e-6)
    return centers, sizes, volumes


def build_spatio_rel_feats(
    tgt_boxes,
    ctx_boxes,
    scene_min_xyz=None,
    scene_whd=None,
    dist_norm_expo=3.0,
    dxdydz_norm_expo=3.0,
    encode_distance='norm_distance',
    encode_orientation='unit_sphere',
    encode_iou='3d_giou',
    encode_size_compare='vol_diff',
    encode_view_dependent='',
    encode_nearest_dist='',
    encode_self=None,
    scene_max_xyz=None,
    all_point_xyz=None,
):
    if encode_self is None:
        encode_self = ['center_norm']
    num_tgt, batch_size, _ = tgt_boxes.shape
    num_ctx, _, _ = ctx_boxes.shape
    tgt_center, tgt_whd, tgt_vol = get_box_property(tgt_boxes)
    ctx_center, ctx_whd, ctx_vol = get_box_property(ctx_boxes)
    tgt_center_norm = (tgt_center - scene_min_xyz[None]) / scene_whd[None]
    ctx_center_norm = (ctx_center - scene_min_xyz[None]) / scene_whd[None]

    union_xyz1 = torch.min(tgt_boxes[:, None, :, :3], ctx_boxes[None, :, :, :3])
    union_xyz2 = torch.max(tgt_boxes[:, None, :, 3:], ctx_boxes[None, :, :, 3:])
    union_whd = (union_xyz2 - union_xyz1).clamp(min=1e-6)
    union_diag = union_whd.norm(p=2, dim=-1, keepdim=True)
    union_bound_vol = union_whd.prod(dim=-1, keepdim=True).clamp(min=1e-6)

    inter_xyz1 = torch.max(tgt_boxes[:, None, :, :3], ctx_boxes[None, :, :, :3])
    inter_xyz2 = torch.min(tgt_boxes[:, None, :, 3:], ctx_boxes[None, :, :, 3:])
    inter_whd = (inter_xyz2 - inter_xyz1).clamp(min=0)
    inter_vol = inter_whd.prod(dim=-1, keepdim=True)

    dxdydz = tgt_center[:, None] - ctx_center[None]
    dxdydz_sign = dxdydz.sign()
    dxdydz_norm = dxdydz / union_whd
    dxdydz_norm = dxdydz_sign * dxdydz_norm.abs().pow(dxdydz_norm_expo)
    dist = dxdydz.norm(p=2, dim=-1, keepdim=True)
    dist_norm = (dist / union_diag).pow(dist_norm_expo)

    union_vol = (tgt_vol[:, None] + ctx_vol[None] - inter_vol).clamp(min=1e-6)
    ious = inter_vol / union_vol
    gious = ious - (union_bound_vol - union_vol) / union_bound_vol

    union_whd_sum = (tgt_whd[:, None] + ctx_whd[None] - inter_whd).clamp(min=1e-6)
    xyz_ious = inter_whd / union_whd
    xyz_gious = xyz_ious - (union_whd - union_whd_sum) / union_whd

    spatial_feats = []
    if encode_distance == 'norm_distance':
        spatial_feats.extend([dxdydz_norm, dist_norm])
    elif encode_distance == 'orig_distance':
        spatial_feats.extend([dxdydz, dist])
    elif encode_distance == 'orig_dxdydz_distxy':
        dist_xy = dxdydz[..., :2].norm(p=2, dim=-1, keepdim=True)
        spatial_feats.extend([dxdydz, dist_xy])
    elif encode_distance == 'orig_dxdydz_distxy_dist':
        dist_xy = dxdydz[..., :2].norm(p=2, dim=-1, keepdim=True)
        spatial_feats.extend([dxdydz, dist_xy, dist])
    elif encode_distance == 'orig_dz_distxy_dist':
        dist_xy = dxdydz[..., :2].norm(p=2, dim=-1, keepdim=True)
        spatial_feats.extend([dxdydz[..., 2:], dist_xy, dist])
    elif encode_distance == 'all_distance':
        spatial_feats.extend([dxdydz, dxdydz_norm, dist, dist_norm])
    elif encode_distance == 'none':
        pass
    else:
        raise NotImplementedError

    if encode_orientation == 'unit_sphere':
        orient_feat = dxdydz / dist.clamp(min=1e-6)
    elif encode_orientation == 'sinθ, cosθ, sinφ, cosφ':
        dist_safe = dist.clamp(min=1e-6)
        dist_xy = dxdydz[..., :2].norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        orient_feat = torch.cat(
            [dxdydz[..., [2]] / dist_safe, dist_xy / dist_safe, dxdydz[..., :2] / dist_xy],
            dim=-1,
        )
    elif encode_orientation == 'none':
        orient_feat = dxdydz.new_zeros([*dxdydz.shape[:-1], 0])
    else:
        orient_feat = torch.atan2(dxdydz[..., [2, 2, 1]], dxdydz[..., [1, 0, 0]]) / np.pi
    spatial_feats.append(orient_feat)

    if encode_iou == '3d_giou':
        spatial_feats.extend([gious])
    elif encode_iou == '3d_giou&iou':
        spatial_feats.extend([gious, ious])
    elif encode_iou == '1d&3d_giou&iou':
        spatial_feats.extend([xyz_gious, xyz_ious, gious, ious])
    elif encode_iou == 'none':
        pass
    else:
        raise NotImplementedError

    if encode_size_compare == 'diff_vol':
        spatial_feats.extend([tgt_vol[:, None] - ctx_vol[None]])
    elif encode_size_compare == 'diff_vol / avg_vol':
        spatial_feats.extend([(tgt_vol[:, None] - ctx_vol[None]) / ((tgt_vol[:, None] + ctx_vol[None]) * 0.5)])
    elif encode_size_compare == 'diff_vol_whd / avg_vol_whd':
        diff_vol = (tgt_vol[:, None] - ctx_vol[None]) / ((tgt_vol[:, None] + ctx_vol[None]) * 0.5)
        diff_whd = (tgt_whd[:, None] - ctx_whd[None]) / ((tgt_whd[:, None] + ctx_whd[None]) * 0.5)
        spatial_feats.extend([diff_vol, diff_whd])
    elif encode_size_compare == 'diff_vol_whd':
        spatial_feats.extend([tgt_vol[:, None] - ctx_vol[None], tgt_whd[:, None] - ctx_whd[None]])
    elif encode_size_compare == 'none':
        pass
    else:
        raise NotImplementedError

    if encode_view_dependent:
        tgt_center_xy = tgt_center[..., :2]
        ctx_center_xy = ctx_center[..., :2]
        compute_scene_center = encode_view_dependent['compute_scene_center']
        encode_view_dep_feat = encode_view_dependent['feat']
        if compute_scene_center == '(max + min) / 2':
            scene_center = (scene_max_xyz + scene_min_xyz) / 2
            scene_center_xy = scene_center[:, :2]
        elif compute_scene_center == '(topk_max + topk_min) / 2':
            topk = encode_view_dependent['k']
            topk_max_xyz = all_point_xyz.topk(k=topk, dim=1, largest=True, sorted=False)[0]
            topk_min_xyz = all_point_xyz.topk(k=topk, dim=1, largest=False, sorted=False)[0]
            scene_center = (topk_max_xyz + topk_min_xyz).mean(dim=1) / 2
            scene_center_xy = scene_center[:, :2]
        elif compute_scene_center == 'inst: (max + min) / 2':
            scene_center = (ctx_center.max(dim=0)[0] + ctx_center.min(dim=0)[0]) / 2
            scene_center_xy = scene_center[:, :2]
        else:
            raise NotImplementedError

        if encode_view_dep_feat == 'ctx2center_as_yaxis':
            yaxis_vec = scene_center_xy[None] - ctx_center_xy
            yaxis_vec = yaxis_vec / yaxis_vec.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            xaxis_vec = torch.cat([-yaxis_vec[..., [1]], yaxis_vec[..., [0]]], -1)
            dxdy = dxdydz[..., :2]
            new_y = (dxdy * yaxis_vec[None]).sum(dim=-1, keepdim=True)
            new_x = (dxdy * xaxis_vec[None]).sum(dim=-1, keepdim=True)
            new_xy = torch.cat([new_x, new_y], dim=-1)
            new_orient = new_xy / new_xy.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            spatial_feats.extend([new_xy, new_orient])
        elif encode_view_dep_feat == 'ctx2center & tgt2center':
            if (tgt_center_xy == ctx_center_xy).all():
                yaxis_vec = scene_center_xy[None] - ctx_center_xy
                yaxis_vec = yaxis_vec / yaxis_vec.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
                xaxis_vec = torch.cat([-yaxis_vec[..., [1]], yaxis_vec[..., [0]]], -1)
                dxdy = dxdydz[..., :2]
                new_y = (dxdy * yaxis_vec[None]).sum(dim=-1, keepdim=True)
                new_x = (dxdy * xaxis_vec[None]).sum(dim=-1, keepdim=True)
                new_xy = torch.cat([new_x, new_y], dim=-1)
                new_orient = new_xy / new_xy.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
                spatial_feats.extend([new_xy, new_orient, new_xy.transpose(0, 1), new_orient.transpose(0, 1)])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    if encode_nearest_dist:
        if encode_nearest_dist != 'nearest_points_on_boxes':
            raise NotImplementedError
        tgt_corner_pt = tgt_center[:, None] - (tgt_whd[:, None] / 2) * dxdydz_sign
        ctx_corner_pt = ctx_center[None] + (ctx_whd[None] / 2) * dxdydz_sign
        nearest_dxdydz = tgt_corner_pt - ctx_corner_pt
        nearest_dxdydz = nearest_dxdydz * (nearest_dxdydz.sign() == dxdydz_sign)
        nearest_dist = nearest_dxdydz.norm(p=2, dim=-1, keepdim=True)
        spatial_feats.extend([nearest_dxdydz, nearest_dist])

    if 'vol' in encode_self:
        spatial_feats.extend([tgt_vol[:, None].expand(-1, num_ctx, -1, -1), ctx_vol[None].expand(num_tgt, -1, -1, -1)])
    if 'center_norm' in encode_self:
        spatial_feats.extend([tgt_center_norm[:, None].expand(-1, num_ctx, -1, -1), ctx_center_norm[None].expand(num_tgt, -1, -1, -1)])
    if 'center_all' in encode_self:
        spatial_feats.extend([
            tgt_center_norm[:, None].expand(-1, num_ctx, -1, -1),
            ctx_center_norm[None].expand(num_tgt, -1, -1, -1),
            tgt_center[:, None].expand(-1, num_ctx, -1, -1),
            ctx_center[None].expand(num_tgt, -1, -1, -1),
        ])
    if 'whd' in encode_self:
        spatial_feats.extend([tgt_whd[:, None].expand(-1, num_ctx, -1, -1), ctx_whd[None].expand(num_tgt, -1, -1, -1)])
    if 'dist2center_horizon' in encode_self:
        dxdydz_to_center = tgt_center - scene_center[None]
        dxdy_to_center = dxdydz_to_center[..., :2]
        dist2center = dxdy_to_center.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        orient2center = dxdy_to_center / dist2center
        spatial_feats.extend([
            dxdy_to_center[:, None].expand(-1, num_ctx, -1, -1),
            dist2center[:, None].expand(-1, num_ctx, -1, -1),
            orient2center[:, None].expand(-1, num_ctx, -1, -1),
        ])
    elif 'tgt_dist2center_horizon' in encode_self:
        dxdydz_to_center = tgt_center - scene_center[None]
        dist2center = dxdydz_to_center[..., :2].norm(p=2, dim=-1, keepdim=True)
        spatial_feats.extend([dist2center[:, None].expand(-1, num_ctx, -1, -1)])
    elif 'both_dist2center_horizon' in encode_self:
        dxdydz_to_center = tgt_center - scene_center[None]
        ctx_dxdydz_to_center = ctx_center - scene_center[None]
        dist2center = dxdydz_to_center[..., :2].norm(p=2, dim=-1, keepdim=True)
        ctx_dist2center = ctx_dxdydz_to_center[..., :2].norm(p=2, dim=-1, keepdim=True)
        spatial_feats.extend([
            dist2center[:, None].expand(-1, num_ctx, -1, -1),
            ctx_dist2center[None].expand(num_tgt, -1, -1, -1),
        ])

    return torch.cat(spatial_feats, dim=-1)


_MODULES = {
    'TransformerEncoderLayer': TransformerEncoderLayer,
    'TransformerDecoder': TransformerDecoder,
    'TransformerDecoderWithRelFeat': TransformerDecoderWithRelFeat,
    'DecoderWithPredHead': DecoderWithPredHead,
    'DecoderBasedMatcherWithRelModel': DecoderBasedMatcherWithRelModel,
    'TransformerDecoderLayer2CrossAttn': TransformerDecoderLayer2CrossAttn,
    'TransformerDecoderLayerWithRelFeat': TransformerDecoderLayerWithRelFeat,
    'TransformerDecoderLayerForRelFeats': TransformerDecoderLayerForRelFeats,
}


def build_layers_from_cfg(layer_cfg, num_layers):
    if isinstance(num_layers, int) and num_layers > 0:
        cfg = layer_cfg.copy()
        layer = _MODULES[cfg.pop('type')](**cfg)
        return _get_clones(layer, num_layers)
    if isinstance(num_layers, list):
        layers = nn.ModuleList()
        for repeat_num, per_layer_cfg in zip(num_layers, layer_cfg):
            cfg = per_layer_cfg.copy()
            layer = _MODULES[cfg.pop('type')](**cfg)
            layers += _get_clones(layer, repeat_num)
        return layers
    return None


CORRELATION_MODULES = {
    'Vis2TextCorrelationLayer': Vis2TextCorrelationLayer,
}


def build_corr_module(cfg):
    corr_cfg = cfg.copy()
    corr_type = corr_cfg.pop('type')
    return CORRELATION_MODULES[corr_type](**corr_cfg)


def build_encoder(cfg):
    enc_cfg = cfg.copy()
    enc_type = enc_cfg.pop('type')
    return _MODULES[enc_type](**enc_cfg)


def build_decoder(cfg):
    dec_cfg = cfg.copy()
    dec_type = dec_cfg.pop('type')
    return _MODULES[dec_type](**dec_cfg)


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu/glu, not {activation}.')
