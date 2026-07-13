#!/bin/bash
#SBATCH --job-name=ecg_train           # 任务名称
#SBATCH --partition=Students           # USTC 107 学生分区
#SBATCH --qos=qos_stu_default          # 学生默认 QoS
#SBATCH --nodes=1                       # 节点数
#SBATCH --ntasks=1                      # 任务数
#SBATCH --cpus-per-task=4               # CPU核心数
#SBATCH --gres=gpu:1                    # 申请1块GPU
#SBATCH --mem=16G                       # 内存（学生分区最大16G）
#SBATCH --time=12:00:00                 # 最大运行时间
#SBATCH --output=logs/%j.out            # 输出日志
#SBATCH --error=logs/%j.err             # 错误日志

# =====================================================
# ECG AI Agent — 模型训练脚本（SLURM 集群提交版）
# =====================================================

set -e  # 遇到错误立即退出

# ---- 0. 环境准备 ----
echo "=== 环境准备 ==="
echo "节点: $(hostname) | 时间: $(date)"

# 显示 GPU 信息
nvidia-smi

# 激活虚拟环境（需要在登录节点提前创建好）
source venv/bin/activate

# 安装依赖（首次运行才需要，后续可注释掉）
pip install --quiet numpy scipy wfdb pandas scikit-learn tqdm matplotlib PyYAML

# 验证 CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# ---- 1. 创建输出目录 ----
mkdir -p logs outputs

# ---- 2. 训练 ----
echo "=== 开始训练 ==="

# 直接做多标签微调（跳过对比预训练，避免AMP NaN问题）
# 对比预训练可选，等微调结果出来后再跑

# 步骤1: 小模型快速验证（InceptionTime, ~30分钟）
echo "--- 1. InceptionTime 基线 ---"
python scripts/train_backbone.py \
    --backbone inception_time \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-4 \
    --num-workers 2 \
    --output-dir outputs/inception_time

# 步骤2: xResNet1D-101 基线（挑战赛冠军方案）
echo "--- 2. xResNet1D-101 基线 ---"
python scripts/train_backbone.py \
    --backbone xresnet1d_101 \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-4 \
    --num-workers 2 \
    --output-dir outputs/xresnet1d_101

# 步骤3: ECG Transformer（创新骨干）
echo "--- 3. ECG Transformer ---"
python scripts/train_backbone.py \
    --backbone ecg_transformer \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 2 \
    --output-dir outputs/ecg_transformer

# ---- 3. 完成 ----
echo "=== 训练完成 ==="
echo "时间: $(date)"
echo "模型保存在: outputs/"
ls -la outputs/
