#!/bin/bash
#SBATCH --job-name=ssl_mask_binning
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/kerryliu/proj_wjk/casanovo/Test_self/logs/slurm_ssl_%j.out
#SBATCH --error=/nfs/turbo/umms-drjieliu/usr/kerryliu/proj_wjk/casanovo/Test_self/logs/slurm_ssl_%j.err
#SBATCH --partition=drjieliu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kerryliu@umich.edu

# =====================================================
# 🧩 环境准备
# =====================================================

# 加载系统模块（CUDA 与 Python）
module load cuda/12.1
module load python/3.10

# -----------------------------------------------------
# 手动桥接到你的 Conda 环境和项目路径
# -----------------------------------------------------
export PYTHONPATH="/nfs/turbo/umms-drjieliu/usr/kerryliu/proj_wjk/casanovo:/home/kerryliu/.conda/envs/casanovo_env/lib/python3.10/site-packages:$PYTHONPATH"

echo "[$(date)] ✅ Environment setup"
echo "Python executable: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

# -----------------------------------------------------
# 检查核心依赖包版本
# -----------------------------------------------------
python - <<'PYCHK'
import sys
print("sys.path =", sys.path[:3])
for pkg in ["numpy", "torch", "pytorch_lightning", "lightning"]:
    try:
        mod = __import__(pkg)
        print(f"✅ {pkg} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"❌ Missing {pkg}: {e}")
PYCHK

# =====================================================
# 📂 进入项目目录并准备输出文件夹
# =====================================================
cd /nfs/turbo/umms-drjieliu/usr/kerryliu/proj_wjk/casanovo/Test_self

mkdir -p logs checkpoints lance_cache

# =====================================================
# 🚀 启动训练
# =====================================================
echo "[$(date)] Starting SSL training..."
python train_ssl_mgf.py
echo "[$(date)] ✅ Training completed."
