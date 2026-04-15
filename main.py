"""Main entry for 3D grounding evaluation."""

import os
import pickle

import torch
import torch.distributed as dist

from evaluation_runtime import BaseEvaluator, init_runtime, parse_args
from models import _MODELS
from models.box_ops import _batch_iou3d_par, box_cxcyczwhd_to_xyzxyz
from src.grounding_dataset import GroundingDataset
from utils.config import Config
from utils.misc import get_rank, get_world_size, is_dist_avail_and_initialized


class GroundingEvaluator(BaseEvaluator):

    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def _build_dataset(args, dataset_dict, split):
        return GroundingDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split=split,
            use_color=args.use_color,
            use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            extra_dataset_cfg=args.extra_dataset_cfg,
            text_encoder_name_or_path=args.text_encoder_name_or_path,
        )

    @staticmethod
    def get_datasets(args):
        dataset_dict = {}
        for dset in args.dataset:
            dataset_dict[dset] = 1
        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        test_dataset = GroundingEvaluator._build_dataset(
            args,
            dataset_dict,
            split='val',
        )
        return None, test_dataset

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        model_args = dict(
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            backbone_checkpoint=args.backbone_checkpoint,
            args=args,
        )
        model_args.update(args.ground_model)
        model = _MODELS[model_args.pop('type')](**model_args)
        return model

    @staticmethod
    def _get_inputs(batch_data):
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            'det_boxes': batch_data['all_detected_boxes'],
            'det_bbox_label_mask': batch_data['all_detected_bbox_label_mask'],
            'det_class_ids': batch_data['all_detected_class_ids'],
            'span_indices': batch_data['span_indices'],
            'expand_indices': batch_data['expand_indices'],
            'span_batch_idx': batch_data['span_batch_idx'],
            'span_batch_ord': batch_data['span_batch_ord'],
            'span_num': batch_data['span_num'],
            'tgt_indices': batch_data['tgt_indices'],
            'is_vgdata': batch_data['is_vgdata'],
            'gt_center': batch_data['center_label'],
            'gt_size': batch_data['size_gts'],
        }
        if 'rand_utterances' in batch_data:
            inputs['rand_text'] = batch_data['rand_utterances']
        if 'span_texts' in batch_data:
            inputs['span_texts'] = batch_data['span_texts']
        if 'noun_texts' in batch_data:
            inputs['noun_texts'] = batch_data['noun_texts']
            inputs['noun_batch_idx'] = batch_data['noun_batch_idx']
            inputs['noun_batch_ord'] = batch_data['noun_batch_ord']
            inputs['noun_num'] = batch_data['noun_num']
            inputs['noun_tgt_indices'] = batch_data['noun_tgt_indices']
        return inputs

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader, model, args):
        """
        Eval grounding after a single epoch.
        """
        del epoch
        model.eval()
        device = next(model.parameters()).device
        accum_acc = 0
        accum_sample = 0
        iou_thrs = torch.as_tensor([0.25, 0.5], device=device)

        ext_stats = {}
        save_test_results = []

        for batch_idx, batch_data in enumerate(test_loader):
            end_points = self._main_eval_branch(batch_idx, batch_data, test_loader, model, args)

            gt_center = end_points['center_label']
            gt_size = end_points['size_gts']
            gt_boxes = torch.cat([gt_center, gt_size], dim=-1)[:, 0]
            num_gt = gt_boxes.size(0)

            pred_boxes = end_points['pred_boxes']
            pred_boxes[..., 3:] = torch.clamp(pred_boxes[..., 3:], min=1e-6)
            bsz = pred_boxes.shape[0]

            pred_logits = end_points['pred_logits']
            if 'pred_logits_s2' in end_points:
                pred_logits_s2 = end_points['pred_logits_s2']
                if pred_logits_s2.shape[1] == pred_logits.shape[1] and 'cand_topk_idx' not in end_points:
                    pred_logits = pred_logits_s2
                else:
                    cand_topk_idx = end_points['cand_topk_idx']
                    pred_logits = torch.full_like(pred_logits, -1e8)
                    pred_logits.scatter_(1, cand_topk_idx, pred_logits_s2)

            _sel_idx = pred_logits.argmax(dim=1)
            batch_ious, _ = _batch_iou3d_par(
                box_cxcyczwhd_to_xyzxyz(pred_boxes),
                box_cxcyczwhd_to_xyzxyz(gt_boxes.unsqueeze(dim=1)),
            )

            preds_iou = batch_ious.gather(
                dim=1,
                index=_sel_idx.reshape(bsz, 1, 1).repeat(1, 1, 1),
            ).reshape(-1)
            accum_acc += (preds_iou[:, None] > iou_thrs[None]).sum(dim=0)
            accum_sample += num_gt

            if args.eval_by_multiple_splits:
                batch_ext_stats = self._eval_by_multiple_splits(batch_data, preds_iou)
                for k, v in batch_ext_stats.items():
                    ext_stats[k] = ext_stats.get(k, 0) + v

            if args.save_test_result_file:
                save_test_results += self._get_save_results(batch_data, end_points, args)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_dist_avail_and_initialized():
            dist.all_reduce(accum_acc)
        accum_sample = torch.tensor(accum_sample, device=accum_acc.device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(accum_sample)
        accum_sample = accum_sample.item()

        if args.eval_by_multiple_splits:
            for k, v in ext_stats.items():
                if isinstance(v, torch.Tensor) and is_dist_avail_and_initialized():
                    dist.all_reduce(v)
                    ext_stats[k] = v.item()
                elif isinstance(v, torch.Tensor):
                    ext_stats[k] = v.item()

        if save_test_results:
            all_save_test_results = [None for _ in range(get_world_size())]
            if is_dist_avail_and_initialized():
                dist.all_gather_object(all_save_test_results, save_test_results)
            else:
                all_save_test_results = [save_test_results]
            if get_rank() == 0:
                all_save_test_results = sum(all_save_test_results, [])
                save_path = os.path.join(args.log_dir, args.save_test_result_file)
                with open(save_path, 'wb') as f:
                    pickle.dump(all_save_test_results, f)
                self.logger.info('Save test results to %s' % save_path)

        for thr, val in zip(iou_thrs.tolist(), accum_acc.tolist()):
            self.logger.info('Acc@%.2f: %.4f' % (thr, val / accum_sample))

        if get_rank() == 0:
            stat_dict = {
                f'Acc@{thr:.2f}': val / accum_sample
                for thr, val in zip(iou_thrs.tolist(), accum_acc.tolist())
            }
            if args.eval_by_multiple_splits:
                split_defs = [
                    ('unique', 'unique_num'),
                    ('multi', 'multi_num'),
                    ('view_dep', 'view_dep_num'),
                    ('view_indep', 'view_indep_num'),
                    ('hard', 'hard_num'),
                    ('easy', 'easy_num'),
                ]
                for split_name, num_key in split_defs:
                    if num_key not in ext_stats:
                        continue
                    split_num = ext_stats[num_key]
                    stat_dict[num_key] = split_num
                    if split_num == 0:
                        continue
                    stat_dict[f'{split_name}_Acc@0.25'] = ext_stats[f'{split_name}_iou_25'] / split_num
                    stat_dict[f'{split_name}_Acc@0.50'] = ext_stats[f'{split_name}_iou_50'] / split_num
            self.logger.info(str(stat_dict))

    def _eval_by_multiple_splits(self, batch_data, preds_iou):
        ext_stats = {}
        if 'is_unique' in batch_data:
            is_unique = batch_data['is_unique'].bool()
            ext_stats['unique_num'] = is_unique.sum()
            ext_stats['unique_iou_25'] = ((preds_iou > 0.25) & is_unique).sum()
            ext_stats['unique_iou_50'] = ((preds_iou > 0.5) & is_unique).sum()
            multi = ~is_unique
            ext_stats['multi_num'] = multi.sum()
            ext_stats['multi_iou_25'] = ((preds_iou > 0.25) & multi).sum()
            ext_stats['multi_iou_50'] = ((preds_iou > 0.5) & multi).sum()
        if 'is_view_dep' in batch_data:
            is_view_dep = batch_data['is_view_dep'].bool()
            ext_stats['view_dep_num'] = is_view_dep.sum()
            ext_stats['view_dep_iou_25'] = ((preds_iou > 0.25) & is_view_dep).sum()
            ext_stats['view_dep_iou_50'] = ((preds_iou > 0.5) & is_view_dep).sum()
            view_indep = ~is_view_dep
            ext_stats['view_indep_num'] = view_indep.sum()
            ext_stats['view_indep_iou_25'] = ((preds_iou > 0.25) & view_indep).sum()
            ext_stats['view_indep_iou_50'] = ((preds_iou > 0.5) & view_indep).sum()
        if 'is_hard' in batch_data:
            is_hard = batch_data['is_hard'].bool()
            ext_stats['hard_num'] = is_hard.sum()
            ext_stats['hard_iou_25'] = ((preds_iou > 0.25) & is_hard).sum()
            ext_stats['hard_iou_50'] = ((preds_iou > 0.5) & is_hard).sum()
            easy = ~is_hard
            ext_stats['easy_num'] = easy.sum()
            ext_stats['easy_iou_25'] = ((preds_iou > 0.25) & easy).sum()
            ext_stats['easy_iou_50'] = ((preds_iou > 0.5) & easy).sum()
        return ext_stats

    def _get_save_results(self, batch_data, end_points, args):
        save_results = []
        save_keys = args.save_test_result_keys
        batch_size = len(batch_data['utterances'])
        for idx in range(batch_size):
            sample = {}
            for key in save_keys:
                if key in batch_data:
                    value = batch_data[key]
                else:
                    value = end_points[key]
                if isinstance(value, torch.Tensor):
                    sample[key] = value[idx].detach().cpu()
                elif isinstance(value, list):
                    sample[key] = value[idx]
                else:
                    sample[key] = value
            save_results.append(sample)
        return save_results

    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model, args):
        return super()._main_eval_branch(batch_idx, batch_data, test_loader, model, args)


def build_args(force_eval=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    opt = parse_args()
    if opt.config:
        cfg = Config(opt.config)
        cfg.merge_to_args(opt)
    opt.eval = True
    if not opt.checkpoint_path:
        raise ValueError('Evaluation requires --checkpoint_path.')
    return opt


def setup_runtime(opt):
    init_runtime(opt)


def run(force_eval=False):
    opt = build_args(force_eval=force_eval)
    setup_runtime(opt)
    evaluator = GroundingEvaluator(opt)
    evaluator.main(opt)


if __name__ == '__main__':
    run()
