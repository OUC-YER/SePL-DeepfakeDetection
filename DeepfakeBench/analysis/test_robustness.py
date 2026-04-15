"""
Robustness Evaluation
========================================
following DeeperForensics-1.0 [CVPR 2020]
using:
    python training/test_robustness.py \
        --detector_path ./training/config/detector/sepl.yaml \
        --weights_path ./weights/ckpt_best.pth \
        --test_dataset FaceForensics++

    #
    python training/test_robustness.py \
        --detector_path ./training/config/detector/sepl.yaml \
        --weights_path ./weights/ckpt_best.pth \
        --test_dataset FaceForensics++ \
        --perturbation_types CS GB GNC
"""

import os
import sys
import json
import random
import argparse
import datetime
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.perturbations import (
    apply_perturbation,
    PERTURBATION_PARAMS,
    PERTURBATION_NAMES,
    DEFAULT_PERTURBATION_TYPES,
)
from detectors import DETECTOR
from metrics.utils import get_test_metrics



class RobustnessTestDataset(DeepfakeAbstractBaseDataset):

    def __init__(self, config, mode='test', perturbation_type=None, severity_level=0):
        super().__init__(config=config, mode=mode)
        self.perturbation_type = perturbation_type
        self.severity_level = severity_level

    def __getitem__(self, index, no_norm=False):
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for i, image_path in enumerate(image_paths):

            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            mask_path = image_path.replace('frames', 'masks')
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                print(f"Error loading image at index {index}: {e}")
                index_random = random.randint(0, len(self.image_list) - 1)
                return self.__getitem__(index_random)

            image = np.array(image)  # RGB, uint8, (H, W, 3)

            if self.perturbation_type is not None and self.severity_level > 0:
                image = apply_perturbation(
                    image, self.perturbation_type, self.severity_level
                )

            # Load mask and landmark
            if self.mode == 'train' and self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None

            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
                if self.config['resolution'] != 256:
                    landmarks = landmarks * (self.config['resolution'] / 256)
            else:
                landmarks = None

            # Test mode
            image_trans = deepcopy(image)
            landmarks_trans = deepcopy(landmarks)
            mask_trans = deepcopy(mask)

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            image_tensors = torch.stack(image_tensors, dim=0)
            if not any(lm is None for lm in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            image_tensors = image_tensors[0]
            if not any(lm is None for lm in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def test_one_dataset(model, data_loader):
    prediction_lists = []
    label_lists = []

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
        data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)

        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())

    return np.array(prediction_lists), np.array(label_lists)



def run_robustness_evaluation(model, config, test_dataset_name,
                              perturbation_types=None, severity_levels=None):

    if perturbation_types is None:
        perturbation_types = DEFAULT_PERTURBATION_TYPES
    if severity_levels is None:
        severity_levels = [1, 2, 3, 4, 5]

    model.eval()
    results = {}

    # ---- Step 0: Clean baseline ----
    print(f"\n{'='*70}")
    print(f"[Clean] 无扰动 baseline - 数据集: {test_dataset_name}")
    print(f"{'='*70}")

    test_config = config.copy()
    test_config['test_dataset'] = test_dataset_name

    clean_dataset = RobustnessTestDataset(
        config=test_config, mode='test',
        perturbation_type=None, severity_level=0
    )
    clean_loader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=int(config['workers']),
        collate_fn=clean_dataset.collate_fn,
        drop_last=False
    )

    preds, labels = test_one_dataset(model, clean_loader)
    clean_metrics = get_test_metrics(y_pred=preds, y_true=labels,
                                     img_names=clean_dataset.data_dict['image'])

    clean_auc = clean_metrics.get('video_auc', clean_metrics.get('auc', 0))
    results['Clean'] = {'level_0': round(clean_auc * 100, 2) if clean_auc <= 1 else round(clean_auc, 2)}
    print(f"  Clean Video-AUC: {results['Clean']['level_0']:.2f}%")

    # ---- Step 1: ----
    for ptype in perturbation_types:
        pname = PERTURBATION_NAMES[ptype]
        results[ptype] = {}
        print(f"\n{'─'*70}")
        print(f"扰动: {pname} ({ptype})")
        print(f"参数: {PERTURBATION_PARAMS[ptype]}")
        print(f"{'─'*70}")

        for level in severity_levels:
            param = PERTURBATION_PARAMS[ptype][level - 1]
            print(f"  Level {level} (param={param})...", end=' ', flush=True)


            perturbed_dataset = RobustnessTestDataset(
                config=test_config, mode='test',
                perturbation_type=ptype, severity_level=level
            )
            perturbed_loader = torch.utils.data.DataLoader(
                dataset=perturbed_dataset,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=perturbed_dataset.collate_fn,
                drop_last=False
            )

            preds, labels = test_one_dataset(model, perturbed_loader)
            metrics = get_test_metrics(y_pred=preds, y_true=labels,
                                       img_names=perturbed_dataset.data_dict['image'])

            auc = metrics.get('video_auc', metrics.get('auc', 0))
            auc_pct = round(auc * 100, 2) if auc <= 1 else round(auc, 2)
            results[ptype][f'level_{level}'] = auc_pct
            print(f"Video-AUC = {auc_pct:.2f}%")

    # ---- Step 2:  Average ----
    results['Average'] = {}
    for level in severity_levels:
        vals = [results[pt][f'level_{level}'] for pt in perturbation_types
                if f'level_{level}' in results.get(pt, {})]
        if vals:
            results['Average'][f'level_{level}'] = round(np.mean(vals), 2)

    return results


def print_results_table(results, perturbation_types=None):

    if perturbation_types is None:
        perturbation_types = DEFAULT_PERTURBATION_TYPES

    severity_levels = [1, 2, 3, 4, 5]

    print(f"\n{'='*80}")
    print(f"鲁棒性实验结果 (Video-level AUC %)")
    print(f"{'='*80}")

    header = f"{'Perturbation':<22}{'Clean':>8}"
    for lv in severity_levels:
        header += f"{'Lv' + str(lv):>8}"
    print(header)
    print("─" * 80)

    for ptype in perturbation_types:
        name = PERTURBATION_NAMES.get(ptype, ptype)
        clean_val = results.get('Clean', {}).get('level_0', '-')
        row = f"{name:<22}{clean_val:>8}"
        for lv in severity_levels:
            val = results.get(ptype, {}).get(f'level_{lv}', '-')
            if isinstance(val, (int, float)):
                row += f"{val:>8.2f}"
            else:
                row += f"{val:>8}"
        print(row)

    # Average
    print("─" * 80)
    clean_val = results.get('Clean', {}).get('level_0', '-')
    row = f"{'Average':<22}{clean_val:>8}"
    for lv in severity_levels:
        val = results.get('Average', {}).get(f'level_{lv}', '-')
        if isinstance(val, (int, float)):
            row += f"{val:>8.2f}"
        else:
            row += f"{val:>8}"
    print(row)
    print("=" * 80)


def plot_robustness_curves(results, perturbation_types=None, output_path='robustness_curves.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not found, skipping plot.")
        return

    if perturbation_types is None:
        perturbation_types = DEFAULT_PERTURBATION_TYPES

    plot_types = perturbation_types + ['Average']
    n_plots = len(plot_types)
    severity_levels = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 2.5, 3.2))
    if n_plots == 1:
        axes = [axes]

    color = '#d62728'
    marker = 'o'

    for ax_idx, ptype in enumerate(plot_types):
        ax = axes[ax_idx]
        pname = PERTURBATION_NAMES.get(ptype, ptype)
        ax.set_title(pname, fontsize=9, fontweight='bold')

        data = results.get(ptype, results.get('Average', {}))
        aucs = [data.get(f'level_{lv}', 0) for lv in severity_levels]

        ax.plot(severity_levels, aucs, color=color, marker=marker,
                linewidth=1.5, markersize=5, label='Ours')

        clean_auc = results.get('Clean', {}).get('level_0', 0)
        ax.axhline(y=clean_auc, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        ax.set_xlabel('Severity', fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel('AUC (%)', fontsize=9)
        ax.set_xticks(severity_levels)
        ax.set_ylim(50, 105)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n鲁棒性曲线图已保存: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Robustness Evaluation (DeeperForensics-1.0 Protocol)')
    parser.add_argument('--detector_path', type=str,
                        default='./training/config/detector/sepl.yaml',
                        help='path to detector YAML config')
    parser.add_argument('--weights_path', type=str,
                        default='./weights/ckpt_best.pth',
                        help='path to model weights')
    parser.add_argument('--test_dataset', type=str, nargs='+',
                        default=['FaceForensics++'],
                        help='test dataset name(s)')
    parser.add_argument('--perturbation_types', type=str, nargs='+',
                        default=None,
                        help='perturbation types to test (default: all 6)')
    parser.add_argument('--output_path', type=str,
                        default='./results/robustness_results.json',
                        help='path to save results JSON')
    parser.add_argument('--plot', action='store_true',
                        help='generate robustness curve plot')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    on_2060 = "2060" in torch.cuda.get_device_name() if torch.cuda.is_available() else False
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['workers'] = 0
    else:
        config['workers'] = 8
        config['lmdb_dir'] = r'/data/disk2/yer/ASOTA/DeepfakeBench/datasets/lmdb'

    config['weights_path'] = args.weights_path

    if config['manualSeed'] is None:
        config['manualSeed'] = 1024
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    np.random.seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
    if config['cudnn']:
        cudnn.benchmark = True

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    if args.weights_path:
        ckpt = torch.load(args.weights_path, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')
            new_weights[new_key] = value
        model.load_state_dict(new_weights, strict=True)
        print('===> Model checkpoint loaded!')
    else:
        print('[ERROR] No weights path provided!')
        return

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    perturbation_types = args.perturbation_types or DEFAULT_PERTURBATION_TYPES

    for pt in perturbation_types:
        assert pt in PERTURBATION_PARAMS, \
            f"Unknown perturbation type '{pt}'. Choose from: {list(PERTURBATION_PARAMS.keys())}"

    print(f"\n{'#'*70}")
    print(f"# 鲁棒性实验 (Robustness Evaluation)")
    print(f"# 遵循 DeeperForensics-1.0 [CVPR 2020] 官方扰动配置")
    print(f"# 模型: {config['model_name']}")
    print(f"# 权重: {args.weights_path}")
    print(f"# 扰动: {perturbation_types}")
    print(f"# 测试集: {args.test_dataset}")
    print(f"{'#'*70}")

    all_results = {}

    for test_name in args.test_dataset:
        print(f"\n{'★'*30} 数据集: {test_name} {'★'*30}")

        results = run_robustness_evaluation(
            model=model,
            config=config,
            test_dataset_name=test_name,
            perturbation_types=perturbation_types,
        )

        all_results[test_name] = results

        print_results_table(results, perturbation_types)

        if args.plot:
            plot_path = args.output_path.replace('.json', f'_{test_name}.png')
            plot_robustness_curves(results, perturbation_types, plot_path)

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    save_data = {
        'timestamp': str(datetime.datetime.now()),
        'model': config['model_name'],
        'weights': args.weights_path,
        'perturbation_types': perturbation_types,
        'perturbation_params': {pt: PERTURBATION_PARAMS[pt] for pt in perturbation_types},
        'results': all_results,
    }

    with open(args.output_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.output_path}")
    print("===> Robustness Test Done!")


if __name__ == '__main__':
    main()