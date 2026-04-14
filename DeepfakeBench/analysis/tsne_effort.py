"""
使用:
  cd /data/disk2/yer/ASOTA/DeepfakeBench
  python training/tsne_effort.py \
      --detector_path ./training/config/detector/effort.yaml \
      --weights_path ./training/weights/ckpt_best.pth \
      --test_dataset FF-real FF-DF FF-F2F FF-FS FF-NT \
      --max_samples 2000 \
      --output_dir ./figures
"""

import os
import sys
import numpy as np
import random
import yaml
import pickle
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI环境下使用

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# === 添加项目路径 ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'training'))

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


# ==================== 参数解析 ====================
parser = argparse.ArgumentParser(description='EFFORT t-SNE Visualization')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/effort.yaml',
                    help='path to detector YAML file')
parser.add_argument('--weights_path', type=str,
                    default='./training/weights/ckpt_best.pth',
                    help='path to model weights')
parser.add_argument('--test_dataset', nargs='+',
                    default=['FF-real', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT'],
                    help='test datasets to use')
parser.add_argument('--max_samples', type=int, default=2000,
                    help='max samples per class (real/fake) for t-SNE')
parser.add_argument('--perplexity', type=float, default=30,
                    help='t-SNE perplexity')
parser.add_argument('--output_dir', type=str, default='./figures',
                    help='output directory for figures')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 工具函数 ====================
def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    """加载配置文件"""
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    # 设置路径
    config['lmdb_dir'] = '/data/disk2/yer/ASOTA/DeepfakeBench/datasets/lmdb'
    config['workers'] = 4
    config['test_dataset'] = args.test_dataset
    config['weights_path'] = args.weights_path

    return config


def prepare_dataloader(config):
    """准备测试数据加载器"""
    test_data_loaders = {}
    for test_name in config['test_dataset']:
        cfg = config.copy()
        cfg['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=cfg, mode='test')
        loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config.get('test_batchSize', 32),
            shuffle=False,
            num_workers=int(config.get('workers', 4)),
            collate_fn=test_set.collate_fn,
            drop_last=False
        )
        test_data_loaders[test_name] = loader
    return test_data_loaders


def load_model(config):
    """加载模型和权重"""
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    ckpt = torch.load(args.weights_path, map_location=device)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    # 去掉 module. 前缀
    new_weights = {}
    for key, value in ckpt.items():
        new_key = key.replace('module.', '')
        new_weights[new_key] = value

    model.load_state_dict(new_weights, strict=True)
    print('===> Model loaded successfully!')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params}')

    return model


# ==================== 特征提取 ====================
@torch.no_grad()
def extract_all_features(model, test_data_loaders):
    """
    提取三种特征: feat_backbone, artifact_feat, content_feat
    以及对应的二分类标签和数据来源标签
    """
    model.eval()

    all_feat_backbone = []
    all_artifact_feat = []
    all_content_feat = []
    all_labels = []          # 二分类标签: 0=Real, 1=Fake
    all_source_labels = []   # 数据来源标签: 用于多类可视化(可选)

    # 数据来源名称到编号的映射
    source_to_id = {}
    source_id_counter = 0

    for dataset_name, loader in test_data_loaders.items():
        # 记录数据来源
        if dataset_name not in source_to_id:
            source_to_id[dataset_name] = source_id_counter
            source_id_counter += 1
        source_id = source_to_id[dataset_name]

        print(f'\n--- Extracting features from: {dataset_name} ---')

        for i, data_dict in tqdm(enumerate(loader), total=len(loader), desc=dataset_name):
            # 准备数据 (与 test.py 一致)
            data = data_dict['image']
            label = data_dict['label']
            mask = data_dict.get('mask', None)
            landmark = data_dict.get('landmark', None)

            # 二分类标签
            label = torch.where(label != 0, 1, 0)

            # 移到 GPU
            data_dict['image'] = data.to(device)
            data_dict['label'] = label.to(device)
            if mask is not None:
                data_dict['mask'] = mask.to(device)
            if landmark is not None:
                data_dict['landmark'] = landmark.to(device)

            # 前向传播
            predictions = model(data_dict, inference=True)

            # 收集三种特征
            all_feat_backbone.append(predictions['feat_backbone'].cpu().numpy())
            all_artifact_feat.append(predictions['artifact_feat'].cpu().numpy())
            all_content_feat.append(predictions['content_feat'].cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_source_labels.extend([source_id] * label.size(0))

    # 拼接所有 batch
    feat_dict = {
        'feat_backbone': np.concatenate(all_feat_backbone, axis=0),
        'artifact_feat': np.concatenate(all_artifact_feat, axis=0),
        'content_feat': np.concatenate(all_content_feat, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
        'source_labels': np.array(all_source_labels),
        'source_to_id': source_to_id,
    }

    print(f'\n===> Total samples: {len(feat_dict["labels"])}')
    print(f'     Real: {(feat_dict["labels"] == 0).sum()}, '
          f'Fake: {(feat_dict["labels"] == 1).sum()}')
    print(f'     Source mapping: {source_to_id}')

    return feat_dict


def balance_and_sample(feat_dict, max_samples_per_class):
    """
    平衡采样: 每类取 max_samples_per_class 个样本
    """
    labels = feat_dict['labels']

    real_indices = np.where(labels == 0)[0]
    fake_indices = np.where(labels == 1)[0]

    n_real = min(len(real_indices), max_samples_per_class)
    n_fake = min(len(fake_indices), max_samples_per_class)
    n_samples = min(n_real, n_fake)  # 取两类中较小的

    sampled_real = np.random.choice(real_indices, size=n_samples, replace=False)
    sampled_fake = np.random.choice(fake_indices, size=n_samples, replace=False)
    sampled_indices = np.concatenate([sampled_real, sampled_fake])
    np.random.shuffle(sampled_indices)

    sampled_dict = {}
    for key in ['feat_backbone', 'artifact_feat', 'content_feat', 'labels', 'source_labels']:
        sampled_dict[key] = feat_dict[key][sampled_indices]
    sampled_dict['source_to_id'] = feat_dict['source_to_id']

    print(f'\n===> Sampled: {n_samples} Real + {n_samples} Fake = {2 * n_samples} total')

    return sampled_dict


# ==================== t-SNE 可视化 ====================
def run_tsne(features, perplexity=30, seed=42):
    """对特征执行 t-SNE 降维"""
    print(f'Running t-SNE (perplexity={perplexity}) on shape {features.shape}...')
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        learning_rate='auto',
        init='pca',
    )
    transformed = tsne.fit_transform(features)
    print('t-SNE done.')
    return transformed


def plot_tsne_three_panel(feat_dict, output_path, perplexity=30, seed=42):
    """
    绘制三张 t-SNE 对比图:
      图1: feat_backbone (解耦前)
      图2: artifact_feat (伪影特征)
      图3: content_feat  (内容特征)
    """
    labels = feat_dict['labels']

    # 定义颜色和标签
    color_map = {0: '#2196F3', 1: '#F44336'}   # 蓝=Real, 红=Fake
    marker_map = {0: '*', 1: 'o'}               # 星=Real, 圆=Fake
    label_name = {0: 'Real', 1: 'Fake'}

    # 三种特征
    feat_keys = ['feat_backbone', 'artifact_feat', 'content_feat']
    titles = [
        '(a) Backbone Features\n(Before Decoupling)',
        '(b) Artifact Features\n(After Decoupling)',
        '(c) Content Features\n(After Decoupling)',
    ]

    # 创建图
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))

    for idx, (feat_key, title) in enumerate(zip(feat_keys, titles)):
        feat = feat_dict[feat_key]

        # 展平特征 (以防万一)
        feat = feat.reshape(feat.shape[0], -1)

        # t-SNE 降维
        transformed = run_tsne(feat, perplexity=perplexity, seed=seed)

        ax = axs[idx]

        # 分别绘制 Real 和 Fake
        for cls in [0, 1]:
            mask = labels == cls
            ax.scatter(
                transformed[mask, 0],
                transformed[mask, 1],
                c=color_map[cls],
                marker=marker_map[cls],
                s=25,
                alpha=0.6,
                label=label_name[cls],
                edgecolors='none',
            )

        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=13, loc='upper right', framealpha=0.9,
                  markerscale=1.5, handletextpad=0.3)

        # 加一个轻微的边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('#CCCCCC')

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n===> Figure saved to: {output_path}')


# ==================== 主函数 ====================
def main():
    init_seed(args.seed)

    # 1. 加载配置
    config = load_config()

    # 2. 加载模型
    model = load_model(config)

    # 3. 准备数据
    test_data_loaders = prepare_dataloader(config)

    # 4. 提取特征
    feat_dict = extract_all_features(model, test_data_loaders)

    # 5. 平衡采样
    feat_dict = balance_and_sample(feat_dict, max_samples_per_class=args.max_samples)

    # 6. 保存特征 (可选, 方便以后复用)
    os.makedirs(args.output_dir, exist_ok=True)
    pkl_path = os.path.join(args.output_dir, 'tsne_features.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(feat_dict, f)
    print(f'Features saved to: {pkl_path}')

    # 7. 绘制 t-SNE 三图对比
    output_path = os.path.join(args.output_dir, 'tsne_effort_decoupling.png')
    plot_tsne_three_panel(
        feat_dict,
        output_path=output_path,
        perplexity=args.perplexity,
        seed=args.seed,
    )

    print('\n===> All done!')


if __name__ == '__main__':
    main()