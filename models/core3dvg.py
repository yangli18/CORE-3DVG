import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
import time

from .backbone_module import Pointnet2Backbone
from .modules import MLP_MODULES, build_module_from_cfg

from .transformer_blocks import build_decoder, build_encoder, build_corr_module, TransformerEncoderLayer

from pointnet2.pointnet2_utils import gather_operation
from .position_embedding import PositionEmbeddingCoordsSine

from .box_ops import box_cxcyczwhd_to_xyzxyz, _batch_iou3d_par
from utils.nms import nms_3d_faster_w_ious
from ops.nms import ml_nms

class Core3DVG(nn.Module):

    def __init__(self, input_feature_dim=3, num_queries=256, num_decoder_layers=6, d_model=288, 
                 backbone_checkpoint=None, init_query_build_method='sample_pt', 
                 final_score_method='default', args=None, extra_cfg={}):
        """Initialize layers."""
        super().__init__()

        del final_score_method
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.args = args

        # Visual encoder
        self.trully_load_backbone_weights = extra_cfg.get('trully_load_backbone_weights', False)
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim, width=1)
        if input_feature_dim == 3 and backbone_checkpoint is not None:
            missing_keys, unexpected_keys = \
            self.backbone_net.load_state_dict(torch.load(backbone_checkpoint, map_location='cpu'), strict=False)
            print(f'missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}')
            if self.trully_load_backbone_weights:
                # import copy
                # backbone_net = copy.deepcopy(self.backbone_net)
                weights = torch.load(backbone_checkpoint, map_location='cpu')['model']
                weight_keys = list(weights.keys())
                for k in weight_keys:
                    if 'sa1.mlp_module.layer0.conv.weight' in k:
                        continue
                    v = weights.pop(k)
                    weights[k.replace('module.backbone_net.', '')] = v
                missing_keys, unexpected_keys = \
                self.backbone_net.load_state_dict(weights, strict=False)
                layer0_conv_weights = weights['module.backbone_net.sa1.mlp_module.layer0.conv.weight']
                # print((self.backbone_net.sa1.mlp_module.layer0.conv.weight[:, :3] != layer0_conv_weights).all())
                # print((self.backbone_net.sa1.mlp_module.layer0.conv.weight == backbone_net.sa1.mlp_module.layer0.conv.weight).sum())
                with torch.no_grad(): # copy the weights for point clouds
                    self.backbone_net.sa1.mlp_module.layer0.conv.weight[:, :3].copy_(layer0_conv_weights)
                print((self.backbone_net.sa1.mlp_module.layer0.conv.weight[:, :3] == layer0_conv_weights).all())
                # print((self.backbone_net.sa1.mlp_module.layer0.conv.weight != backbone_net.sa1.mlp_module.layer0.conv.weight).sum())
                print(f'missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}')
            # for k, v in self.backbone_net.named_parameters():
            #     if not (backbone_net.state_dict()[k] == self.backbone_net.state_dict()[k]).all():
            #         print(k, ': not the same')

        # Text Encoder
        t_type = args.text_encoder_name_or_path
        for _ in range(50):
            try:
                if extra_cfg.get('bert_config', {}):
                    _config = RobertaConfig.from_pretrained(t_type)
                    _config.update(extra_cfg.get('bert_config', {}))
                    self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
                    self.text_encoder = RobertaModel.from_pretrained(t_type, config=_config)
                else:
                    self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
                    self.text_encoder = RobertaModel.from_pretrained(t_type)
                break
            except:
                time.sleep(1)
                continue
        if args.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        for v in self.text_encoder.pooler.parameters():
            v.requires_grad_(False)

        self.text_encoder.config.output_hidden_states = True
        
        self.text_max_len = extra_cfg.get('text_max_len', None)

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        if extra_cfg.get('decoder_s2_text_feat') != 'proj_text_hs':
            raise ValueError('Only supports decoder_s2_text_feat="proj_text_hs".')
        self.text_projector_s2 = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # Encoder
        assert isinstance(args.encoder, dict)
        if args.encoder:
            self.encoder = build_encoder(args.encoder)
        else:
            self.encoder = None

        # Decoders
        assert isinstance(args.decoder, dict)
        self.decoder_stage1 = build_decoder(args.decoder['decoder_stage1'])
        self.decoder_stage2 = build_decoder(args.decoder['decoder_stage2'])

        self.decoder_stage2_other_scores = extra_cfg.get('decoder_stage2_other_scores', [])
        if self.decoder_stage2_other_scores not in ([], ['rel_pred_logits']):
            raise ValueError('Only supports decoder_stage2_other_scores=["rel_pred_logits"].')

        # point-cloud position embeddding
        if args.point_cloud_pos_embed:
            self.pt_pos_embed = PositionEmbeddingCoordsSine(**args.point_cloud_pos_embed)
        else:
            self.pt_pos_embed = None
        
        self.pt_pos_mlp = build_module_from_cfg(MLP_MODULES, args.point_cloud_pos_mlp)

        if init_query_build_method != 'sample_pt':
            raise ValueError('Only supports init_query_build_method="sample_pt".')
        self.sample_pt_method = args.sample_pointcloud['method']
        if self.sample_pt_method != 'topk_corr_score':
            raise ValueError('Only supports sample_pointcloud.method="topk_corr_score".')
        self.v2t_corr = build_corr_module(extra_cfg.get('v2t_corr_cfg'))

        self.sample_pt_num = args.sample_pointcloud['num']
        if args.sample_pointcloud.get('scale_num', 1) != 1:
            raise ValueError('Only supports sample_pointcloud.scale_num=1.')
        self.sample_pt_pred_head = build_module_from_cfg(MLP_MODULES, args.sample_pointcloud['pred_head'])
        self.sample_pt_scores = args.sample_pointcloud.get('sample_scores', False)

        assert isinstance(args.decoder_extra_operation, dict)
        self.decoder_stage1_extra_ops = args.decoder_extra_operation['decoder_stage1_extra_ops']
        self.decoder_stage2_extra_ops = args.decoder_extra_operation['decoder_stage2_extra_ops']
        if [op['method'] for op in self.decoder_stage1_extra_ops] != ['decode_noun_text']:
            raise ValueError('Only supports decoder_stage1_extra_ops=[decode_noun_text].')
        if [op['method'] for op in self.decoder_stage2_extra_ops] != ['decode_rand_text']:
            raise ValueError('Only supports decoder_stage2_extra_ops=[decode_rand_text].')

        self.noun_span_type = extra_cfg.get('noun_span_type', 'default')
        if self.noun_span_type != 'default':
            raise ValueError('Only supports noun_span_type="default".')

        candidate_select = extra_cfg.get('candidate_select', {})
        self.use_gpu_nms = candidate_select.get('use_gpu_nms', False)
        self.topk_cand = candidate_select.get('topk_cand', 50)
        self.topk_cand_by_nms = candidate_select.get('topk_cand_by_nms', False)
        self.topk_cand_nms_iou = candidate_select.get('topk_cand_nms_iou', 0.6)
        if not self.topk_cand_by_nms:
            raise ValueError('Only supports candidate selection with NMS.')

        # Init weights
        self.init_bn_momentum()


    def get_point_pos_embeddings(self, point_xyz, point_range=None):
        if self.pt_pos_embed:
            point_pos_embed = self.pt_pos_embed(point_xyz, input_range=point_range) # [bs, c, #point]
            point_pos_embed = point_pos_embed.permute(0, 2, 1)
        else:
            point_pos_embed = point_xyz
        point_pos_embed = self.pt_pos_mlp(point_pos_embed)
        return point_pos_embed


    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1


    def run_text_encoder(self, text, device):
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="longest", return_tensors="pt", max_length=self.text_max_len
        ).to(device) # 'longest' means padding to the longest length in the batch
        encoded_text = self.text_encoder(**tokenized)
        text_hs = encoded_text.last_hidden_state
        text_feats = encoded_text.last_hidden_state
        text_padding_mask = tokenized.attention_mask.ne(1).bool()
        return text_feats, text_padding_mask, text_hs, tokenized


    def run_backbones(self, inputs):
        """Run visual and text backbones."""
        # Visual encoder
        end_points = self.backbone_net(inputs['point_clouds'], end_points={})
        device = inputs['point_clouds'].device
        
        # Text encoder
        text_feats, text_padding_mask, text_hs, tokenized = \
            self.run_text_encoder(inputs['text'], device)
        text_feats = self.text_projector(text_feats)
        end_points['text_feats'] = text_feats
        end_points['text_padding_mask'] = text_padding_mask
        end_points['text_hidden_states'] =  text_hs
        end_points['tokenized'] = tokenized

        if 'span_indices' in inputs:
            span_batch_idx = inputs['span_batch_idx']
            span_batch_ord = inputs['span_batch_ord']
            span_num = inputs['span_num']
            tgtspan_idx = inputs['tgt_indices']
            span_indices = inputs['span_indices']
            
            first_tgtspan_idx = tgtspan_idx[:, 0]
            has_span_mask = first_tgtspan_idx >= 0
            tgtspan_idx = tgtspan_idx[has_span_mask]
            span_num = span_num[has_span_mask]
            _mask = has_span_mask[span_batch_idx]
            span_batch_idx = span_batch_idx[_mask]
            span_batch_ord = span_batch_ord[_mask]
            span_indices = span_indices[_mask]

            end_points['span_batch_idx'] = span_batch_idx
            end_points['span_batch_ord'] = span_batch_ord
            end_points['has_span_mask'] = has_span_mask
            end_points['span_num'] = span_num
            end_points['tgtspan_idx'] = tgtspan_idx

        if 'noun_texts' in inputs:
            noun_texts = inputs['noun_texts']
            noun_texts = [t for texts in noun_texts for t in texts.split('||')]

            noun_text_feats, noun_text_padding_mask, _, _ = \
                self.run_text_encoder(noun_texts, device)
            noun_text_feats = self.text_projector(noun_text_feats)
            end_points.update({
                "noun_text_feats": noun_text_feats,
                "noun_text_padding_mask": noun_text_padding_mask,
            })

        if 'rand_text' in inputs:
            rand_text = inputs['rand_text']
            rand_text_feats, rand_text_padding_mask, rand_text_hs, _ = \
                self.run_text_encoder(rand_text, device)
            rand_text_feats = self.text_projector(rand_text_feats)
            end_points.update({
                "rand_text_hidden_states": rand_text_hs,
                "rand_text_feats": rand_text_feats,
                "rand_text_padding_mask": rand_text_padding_mask,
            })

        return end_points
    

    def forward(self, inputs):
        end_points = self.run_backbones(inputs)
        point_xyz = end_points['fp2_xyz']  # (B, points, 3)
        point_feats = end_points['fp2_features']  # (B, F, points)
        text_feats = end_points['text_feats']  # (B, L, F)
        text_padding_mask = end_points['text_padding_mask']  # (B, L)
        B, _, _ = point_feats.shape

        # Prepare the point clouds' feats and pos
        point_feats_BxCxN = point_feats
        point_feats = point_feats.permute(2, 0, 1) # [#point, B, C]
        text_feats = text_feats.transpose(0, 1) # [L, B, C]

        point_pos = self.get_point_pos_embeddings(point_xyz, None) # [B, #point, c]
        point_pos = point_pos.transpose(0, 1)  # [#point, B, c]

        ext_input = dict()
        # ext_input['positive_map'] = inputs['positive_map']

        if self.encoder is not None:
            if isinstance(self.encoder, TransformerEncoderLayer):
                point_feats = self.encoder(point_feats, pos=point_pos)
            else:
                raise
        
        corr_scores = self.v2t_corr(point_feats, point_pos, text_feats, text_padding_mask)
        end_points['point_corr_scores'] = corr_scores.transpose(0, 1)
        end_points['point_xyz'] = point_xyz
        sample_inds = corr_scores.transpose(0, 1).reshape(B, -1).topk(k=self.sample_pt_num, dim=-1)[1].int()
        if self.sample_pt_scores:
            sample_pt_scores = gather_operation(corr_scores.permute(1, 2, 0).contiguous(), sample_inds)
            end_points['sample_pt_scores'] = sample_pt_scores.transpose(1, 2).contiguous()

        prop_boxes = None
        prop_padding_mask = None
        xyz_flipped = point_xyz.transpose(1, 2).contiguous()
        sample_pt_xyz = gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        sample_pt_feat = gather_operation(point_feats_BxCxN, sample_inds).transpose(1, 2).contiguous()
        sample_pt_boxes = self.sample_pt_pred_head(sample_pt_feat)
        sample_pt_boxes[..., :3] = sample_pt_boxes[..., :3] + sample_pt_xyz
        end_points['sample_pt_xyz'] = sample_pt_xyz
        end_points['sample_pt_boxes'] = sample_pt_boxes

        query_mask = torch.ones((B, self.sample_pt_num), device=sample_pt_boxes.device, dtype=torch.bool)
        end_points['query_mask'] = query_mask


        '''---------------------'''
        '''| 1st-stage decoder |'''
        '''---------------------'''
        dec_res_s1 = self.decoder_stage1(point_feats, point_pos, 
                                text_feats, text_padding_mask, 
                                prop_boxes, prop_padding_mask,
                                sample_pt_feat, sample_pt_boxes, sample_pt_xyz,
                                ext_input) # [#layer, #query, bs, c]

        # The predicted logits & boxes
        pred_logits = dec_res_s1['pred_logits'].transpose(1, 2)# [#layer, B, Q, 1]
        pred_logits = pred_logits.where(query_mask[None, ..., None], pred_logits.new_tensor(-100000000))
        pred_boxes = dec_res_s1['pred_boxes'].transpose(1, 2) # [#layer, B, Q, 6]
        pred_box_mask = dec_res_s1['pred_box_mask'] # [#layer, B, Q]

        end_points['pred_logits'] = pred_logits[-1] # [B, Q, 1]
        end_points['pred_boxes'] = pred_boxes[-1] # [B, Q, 6]
        end_points['pred_box_mask'] = pred_box_mask[-1] # [B, Q]
        end_points['query_mask'] = query_mask # [B, Q]
        
        if self.args.aux_outputs:
            end_points['aux_outputs'] = [{'pred_boxes': b, 'pred_logits': p, 'pred_box_mask':m}
                                        for b, p, m in zip(pred_boxes[:-1], pred_logits[:-1], pred_box_mask[:-1])]
        if 'align_logits' in dec_res_s1:
            init_align_logits = dec_res_s1['init_align_logits']
            align_logits = dec_res_s1['align_logits'] # [#dec_layer, B, Q, T]
            end_points['init_align_logits'] = init_align_logits # [B, Q, T]
            end_points['align_logits'] = align_logits[-1] # [B, Q, T]
            if self.args.aux_outputs:
                for aux_out, p in zip(end_points['aux_outputs'], align_logits[:-1]):
                    aux_out['align_logits'] = p

        if 'init_boxes' in dec_res_s1:
            end_points['init_boxes'] = dec_res_s1['init_boxes'].transpose(0, 1) # [B, Q, 6]

        is_vgdata = inputs['is_vgdata']
        noun_batch_idx = inputs['noun_batch_idx']
        noun_batch_ord = inputs['noun_batch_ord']
        noun_text_feats = end_points['noun_text_feats'].transpose(0, 1)
        noun_text_padding_mask = end_points['noun_text_padding_mask']
        noun_dec_res = self.decoder_stage1(
            point_feats[:, noun_batch_idx], point_pos[:, noun_batch_idx],
            noun_text_feats, noun_text_padding_mask,
            None, None, sample_pt_feat[noun_batch_idx],
            sample_pt_boxes[noun_batch_idx], sample_pt_xyz[noun_batch_idx],
            ext_input)

        noun_pred_logits = noun_dec_res['pred_logits'].transpose(1, 2)
        noun_pred_boxes = noun_dec_res['pred_boxes'].transpose(1, 2)
        noun_tgt_idx = inputs['noun_tgt_indices'].clone()
        max_tgt_idx = noun_tgt_idx.max(-1)[0][noun_batch_idx]
        noun_tgt_idx[noun_tgt_idx == -1] = 1e6
        min_tgt_idx = noun_tgt_idx.min(-1)[0][noun_batch_idx]
        has_label = (noun_batch_ord >= min_tgt_idx) & (noun_batch_ord <= max_tgt_idx)
        has_label = has_label | (~is_vgdata[noun_batch_idx])

        noun_outputs_wlabel = {
            'pred_logits': noun_pred_logits[:, has_label][-1],
            'pred_boxes': noun_pred_boxes[:, has_label][-1],
            'noun_batch_idx': noun_batch_idx[has_label],
            'noun_batch_ord': noun_batch_ord[has_label],
        }
        noun_outputs_wolabel = {
            'pred_logits': noun_pred_logits[:, ~has_label][-1],
            'pred_boxes': noun_pred_boxes[:, ~has_label][-1],
            'noun_batch_idx': noun_batch_idx[~has_label],
            'noun_batch_ord': noun_batch_ord[~has_label],
        }
        if self.args.aux_outputs:
            noun_outputs_wlabel['aux_outputs'] = [
                {'pred_boxes': b, 'pred_logits': p}
                for b, p in zip(noun_pred_boxes[:, has_label][:-1], noun_pred_logits[:, has_label][:-1])
            ]
            noun_outputs_wolabel['aux_outputs'] = [
                {'pred_boxes': b, 'pred_logits': p}
                for b, p in zip(noun_pred_boxes[:, ~has_label][:-1], noun_pred_logits[:, ~has_label][:-1])
            ]

        end_points['noun_outputs_wlabel'] = noun_outputs_wlabel
        end_points['noun_outputs_wolabel'] = noun_outputs_wolabel



        '''----------------'''
        '''| 2nd: matcher |'''
        '''----------------'''
        cand_logits = pred_logits[-1]
        cand_boxes = pred_boxes[-1]
        cand_boxes_xyzxyz = box_cxcyczwhd_to_xyzxyz(cand_boxes)
        cand_logits_for_nms = cand_logits
        if self.use_gpu_nms:
            _, Q, _ = cand_boxes.shape
            cand_batchidx = torch.arange(B).repeat_interleave(Q).to(cand_boxes.device)
            keep = ml_nms(cand_boxes_xyzxyz.view(-1, 6), cand_logits_for_nms.view(-1),
                            cand_batchidx, self.topk_cand_nms_iou)
            keeped_cand_logits = torch.full_like(cand_logits_for_nms, -1e6)
            keeped_cand_logits[keep // Q, keep % Q] = cand_logits_for_nms.view(-1, 1)[keep]
            cand_logits, cand_topk_idx = keeped_cand_logits.topk(k=self.topk_cand, dim=1)
            cand_keep_mask = (cand_logits > -1e6).squeeze(-1)
        else:
            sort_idxes = cand_logits_for_nms.squeeze(-1).argsort(dim=1).cpu().numpy()
            cand_ious = _batch_iou3d_par(cand_boxes_xyzxyz, cand_boxes_xyzxyz)[0].detach().cpu().numpy()
            cand_topk_idx = [nms_3d_faster_w_ious(sort_idx, iou, self.topk_cand_nms_iou, self.topk_cand)
                                for sort_idx, iou in zip(sort_idxes, cand_ious)]
            cand_topk_idx = cand_logits.new_tensor(cand_topk_idx, dtype=torch.long).view(B, -1, 1)
            cand_logits = cand_logits.gather(1, cand_topk_idx)
            
        
        cand_queries = dec_res_s1['tgt_queries'] # [Q, B, C]
        C = cand_queries.shape[-1]
        cand_queries = cand_queries.gather(0, cand_topk_idx.transpose(0, 1).expand(-1, -1, C)) # [N1, B, C]
        cand_boxes = cand_boxes.gather(1, cand_topk_idx.expand(-1, -1, 6)) # [B, N1, 6]
        cand_query_mask = end_points['query_mask'].gather(1, cand_topk_idx.view(B, -1)) # [B, N1]
        if self.use_gpu_nms:
            cand_query_mask = cand_query_mask & cand_keep_mask
        cand_query_padding_mask = ~cand_query_mask
        K, _, _ = cand_queries.shape
        end_points['cand_logits'] = cand_logits # [B, N1, 1]
        end_points['cand_boxes'] = cand_boxes # [B, N1, 6]
        end_points['cand_query_mask'] = cand_query_mask # [B, N1]
        end_points['cand_topk_idx'] = cand_topk_idx # [B, N1, 1]

        text_hs = end_points['text_hidden_states']
        text_feats_s2 = self.text_projector_s2(text_hs).transpose(0, 1)


        # cand_queries: [K, B, C]
        # cand_boxes: [B, K, 6] -> [K, B, 6]
        # cand_query_padding_mask: [B, K]   
        cand_boxes = cand_boxes.transpose(0, 1) # [K, B, 6]
        ext_input['all_point_xyz'] = inputs['point_clouds'][..., :3]
        ext_input['point_feats'] = point_feats
        ext_input['point_feat_pos'] = point_pos
        dec_res_s2 = self.decoder_stage2(cand_queries, cand_boxes.detach(),
                                        text_feats_s2, text_padding_mask, 
                                        cand_queries, cand_boxes.detach(), cand_query_padding_mask,
                                        ext_input) # [#layer, #query, bs, c]

        # The predicted logits
        pred_logits_s2 = dec_res_s2['pred_logits'].transpose(1, 2)# [#layer, B, Q2, 1]
        pred_logits_s2 = pred_logits_s2.where(cand_query_mask[None, ..., None], pred_logits.new_tensor(-1e6))
        end_points['pred_logits_s2'] = pred_logits_s2[-1] # [B, Q, 1]
        if self.args.aux_outputs and pred_logits_s2.shape[0] > 1:
            end_points['aux_outputs_s2'] = [{'pred_logits_s2': p} for p in pred_logits_s2[:-1]]

        if self.decoder_stage2_other_scores:
            rel_pred_logits = dec_res_s2['rel_pred_logits'].transpose(1, 2)
            rel_pred_logits = rel_pred_logits.where(cand_query_mask[None, ..., None], rel_pred_logits.new_tensor(-1e6))
            end_points['rel_pred_logits'] = rel_pred_logits[-1]
            if self.args.aux_outputs and rel_pred_logits.shape[0] > 1:
                for aux_out, p in zip(end_points['aux_outputs_s2'], rel_pred_logits[:-1]):
                    aux_out['rel_pred_logits'] = p

        rand_text_hs = end_points['rand_text_hidden_states']
        rand_text_feats_s2 = self.text_projector_s2(rand_text_hs).transpose(0, 1)
        rand_text_padding_mask = end_points['rand_text_padding_mask']
        if hasattr(self.decoder_stage2, 'forward_rand_text'):
            rand_dec_res_s2 = self.decoder_stage2.forward_rand_text(
                cand_queries, cand_boxes.detach(),
                rand_text_feats_s2, rand_text_padding_mask,
                cand_queries, cand_boxes.detach(), cand_query_padding_mask, ext_input
            )
        else:
            rand_dec_res_s2 = self.decoder_stage2(
                cand_queries, cand_boxes.detach(),
                rand_text_feats_s2, rand_text_padding_mask,
                cand_queries, cand_boxes.detach(), cand_query_padding_mask, ext_input
            )
        rand_logits_s2 = rand_dec_res_s2['pred_logits'].transpose(1, 2)
        end_points['rand_logits_s2'] = rand_logits_s2[-1]
        if self.args.aux_outputs and rand_logits_s2.shape[0] > 1:
            for aux_out, p in zip(end_points['aux_outputs_s2'], rand_logits_s2[:-1]):
                aux_out['rand_logits_s2'] = p

        return end_points
