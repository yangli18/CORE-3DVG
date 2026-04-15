import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from utils.logger import setup_logger
from utils.misc import collate_fn_w_span_indices
from utils.misc import get_rank, is_dist_avail_and_initialized


def parse_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser()

    # Model / runtime behavior
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number')
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--detect_intermediate', action='store_true')

    # Data
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, default=['sr3d'], nargs='+')
    parser.add_argument('--test_dataset', default='sr3d')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true')
    parser.add_argument('--use_color', action='store_true')
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--extra_dataset_cfg', default=dict())

    # IO
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path')
    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--text_encoder_name_or_path', default='roberta-base')

    # Runtime
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--rng_seed', type=int, default=0, help='manual seed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--backbone_checkpoint', default=None)

    # Config payload
    parser.add_argument('--config', type=str)
    parser.add_argument('--ground_model', type=dict, default={})
    parser.add_argument('--encoder', default={})
    parser.add_argument('--decoder', default={})
    parser.add_argument('--decoder_extra_operation', default={})
    parser.add_argument('--matcher', default=dict())
    parser.add_argument('--matcher_extra_operation', default={})
    parser.add_argument('--proposal_select', default=dict())
    parser.add_argument('--point_cloud_pos_embed', type=dict)
    parser.add_argument('--point_cloud_pos_mlp', type=dict)
    parser.add_argument('--separate_pred_heads', default=False)
    parser.add_argument('--pred_head', default=False)

    # Loss config
    parser.add_argument('--freeze_text_encoder', default=True)
    parser.add_argument('--pred_heads_in_decoder', default=False)
    parser.add_argument('--aux_outputs', default=True)
    parser.add_argument('--sample_pointcloud', default=dict())

    # Eval outputs
    parser.add_argument(
        '--save_test_result_keys',
        default=[
            'scan_ids', 'utterances', 'span_batch_idx', 'span_batch_ord',
            'span_num', 'decomp_logits', 'init_boxes', 'pred_logits',
            'pred_boxes', 'aux_outputs'
        ],
    )
    parser.add_argument('--save_test_result_file', default='')
    parser.add_argument('--use_eval_dir', action='store_true')
    parser.add_argument('--eval_by_multiple_splits', action='store_true')

    args, _ = parser.parse_known_args()
    args.eval = True
    return args


def load_checkpoint(args, model, checkpoint_path=None):
    """Load model weights from checkpoint."""
    if not checkpoint_path:
        checkpoint_path = args.checkpoint_path
    print("=> loading model weights '{}'".format(checkpoint_path))

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if not (isinstance(state_dict, dict) and all(torch.is_tensor(v) for v in state_dict.values())):
        raise KeyError('Unsupported checkpoint format.')
    target_model = model.module if isinstance(model, DistributedDataParallel) else model
    try:
        target_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            target_model.load_state_dict(state_dict, strict=True)
        else:
            raise
    print("=> loaded model weights successfully '{}'".format(checkpoint_path))

    del state_dict
    torch.cuda.empty_cache()


def init_runtime(args):
    """Initialize single-process or distributed runtime."""
    use_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    args.distributed = use_distributed
    if use_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.enabled = True
    return args


class BaseEvaluator:
    """Basic evaluation class to be inherited."""

    def __init__(self, args):
        name = args.log_dir.split('/')[-1]
        if args.use_eval_dir:
            args.log_dir = os.path.join(args.log_dir, ','.join(args.dataset), 'eval')
        else:
            args.log_dir = os.path.join(args.log_dir, ','.join(args.dataset), f'{int(time.time())}')
        os.makedirs(args.log_dir, exist_ok=True)

        self.logger = setup_logger(
            output=args.log_dir,
            distributed_rank=get_rank(),
            name=name,
        )
        if get_rank() == 0:
            path = os.path.join(args.log_dir, 'config.json')
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

    @staticmethod
    def get_datasets(args):
        del args
        return None, None

    def get_test_loader(self, args):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        _, test_dataset = self.get_datasets(args)
        g = torch.Generator()
        g.manual_seed(0)
        if args.distributed:
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            shuffle = False
        else:
            test_sampler = SequentialSampler(test_dataset)
            shuffle = False
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            collate_fn=collate_fn_w_span_indices,
            sampler=test_sampler,
            drop_last=False,
            generator=g,
        )
        return test_loader

    @staticmethod
    def get_model(args):
        del args
        return None

    def main(self, args):
        """Run the evaluation pipeline."""
        model = self.get_model(args)

        if torch.cuda.is_available():
            model = model.cuda()
        if args.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                broadcast_buffers=False,
            )

        test_loader = self.get_test_loader(args)
        self.logger.info(f"length of testing dataset: {len(test_loader.dataset)}")

        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model)

        print("Testing evaluation.....................")
        self.evaluate_one_epoch(0, test_loader, model, args)

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
        }

    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model, args):
        del batch_idx, test_loader
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        inputs['train'] = False
        end_points = model(inputs)
        end_points.update(**batch_data)
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)
        return end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader, model, args):
        del epoch, test_loader, model, args
        return None
