import math
import torch
import torch.nn as nn
import pytorch_lightning as pl #import lightning.pytorch as pl
import torch.nn.functional as F
from casanovo.denovo.transformers import SpectrumEncoder
import matplotlib.pyplot as plt
import os
from typing import List, Dict
# 改预测目标 改成m/z 删掉余弦 
# 读DreaMS/casanovo encoder
class SpectrumSSL(pl.LightningModule):
    """
    Self-supervised spectrum modeling (encoder-only).
    Curriculum masking + binning + cross-entropy loss.
    Now masks m/z values instead of intensities and predicts m/z values.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        lr: float = 1e-4,
        #n_bins: int = 64,
        min_mask: float = 0.05,
        max_mask: float = 0.5,
        total_epochs: int = 100,
        use_cosine_mask_schedule: bool = False,
        mz_min: float = 100.0,
        mz_max: float = 2000.0,
        log_scale: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize lists to store losses for plotting
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.current_epoch_losses: List[float] = []

        # 基本参数初始化
        self.lr = lr
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.log_scale = log_scale
        self.MASK_TOKEN = -1.0
        
        # 设置bin相关参数
        self.bin_size = 0.1  # 固定bin大小为0.1
        self.n_bins = int((self.mz_max - self.mz_min) / self.bin_size)  # 根据mz范围和bin大小计算bin数量
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()  # 这里有softmax层所以训练阶段就不用了

        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        # MLP
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, self.n_bins),  # 使用self.n_bins而不是n_bins
        )

    # Mask rate scheduler
    def current_mask_rate(self):
        epoch = self.current_epoch
        min_mask = self.hparams.min_mask    # 最小 mask 比例
        max_mask = self.hparams.max_mask    # 最大 mask 比例
        total_epochs = max(1, self.hparams.total_epochs)
    # 余弦调度
        if self.hparams.use_cosine_mask_schedule:
            rate = max_mask - (max_mask - min_mask) * 0.5 * (
                1 + math.cos(math.pi * epoch / total_epochs)
            )
        else:
            rate = min_mask + (max_mask - min_mask) * (epoch / total_epochs)
        return float(min(rate, max_mask))

    # Discretize m/z values 连续m/z值转成分类标签
    def bin_mz(self, mzs: torch.Tensor) -> torch.Tensor:
        """
        把 m/z 映射到 [0, n_bins-1] 的整数类别。
        借鉴selfsupervise.py中的mz_to_bin函数
        """
        # 避免全 0 pad 引发 NaN
        valid_mask = mzs > 0
        if not valid_mask.any():
            return torch.zeros_like(mzs, dtype=torch.long)

        # Use a fixed bin width instead of dynamic scaling between mz_min/mz_max.
        # Minimal change: choose a fixed interval (example: 0.1) and compute bin index
        # as floor((mz - mz_min) / bin_width). Clamp to [0, n_bins-1].
        bin_width = 0.1

        # Ensure we do not produce negative bins for values < mz_min by clamping
        clamped = torch.clamp(mzs, min=self.mz_min)
        bins = torch.floor((clamped - self.mz_min) / bin_width).long()

        # Clamp to valid range and set invalid positions to 0
        bins = torch.clamp(bins, min=0, max=self.n_bins - 1)
        bins[~valid_mask] = 0
        return bins.detach()

    # Masking function 随机掩盖一部分峰的m/z值（mask）
    def mask_spectrum(self, mzs: torch.Tensor, intensities: torch.Tensor, mask_rate: float):
        """
        借鉴selfsupervise.py的掩码策略，对m/z值进行掩码
        """
        B, L = mzs.shape
        # 只对有效的峰（m/z > 0）进行掩码
        valid_mask = mzs > 0
        mask = torch.zeros_like(mzs, dtype=torch.bool)
        
        for b in range(B):
            valid_indices = torch.where(valid_mask[b])[0]
            if len(valid_indices) > 0:
                n_mask = max(1, int(mask_rate * len(valid_indices)))
                chosen_indices = valid_indices[torch.randperm(len(valid_indices))[:n_mask]]
                mask[b, chosen_indices] = True
        
        masked_mzs = mzs.clone()
        
        # 应用掩码策略：80%用MASK_TOKEN，10%用随机值，10%保持原值
        for b in range(B):
            masked_positions = torch.where(mask[b])[0]
            for pos in masked_positions:
                r = torch.rand(1).item()
                if r < 0.8:  # 80%概率：用MASK_TOKEN替换
                    masked_mzs[b, pos] = self.MASK_TOKEN
                elif r < 0.9:  # 10%概率：用随机值替换
                    if self.log_scale:
                        masked_mzs[b, pos] = torch.exp(
                            torch.rand(1) * (math.log(self.mz_max) - math.log(self.mz_min)) + math.log(self.mz_min)
                        ).item()
                    else:
                        masked_mzs[b, pos] = torch.rand(1).item() * (self.mz_max - self.mz_min) + self.mz_min
                # else: 10%概率保持原值不变
        
        return masked_mzs, mask

    # Forward encoder➕MLP
    def forward(self, mzs, intensities):
        enc, _ = self.encoder(mzs, intensities)
        logits = self.mlp_head(enc)  # (B, L, n_bins)
        return logits

    # Training step
    def training_step(self, batch, _):
        mzs = batch["mz_array"]
        intensities = batch["intensity_array"]
        #mask_rate = self.current_mask_rate()
        mask_rate = 0.05
        labels = self.bin_mz(mzs)
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate)

        logits = self(masked_mzs, intensities)  # (B, L, n_bins)
        # 换成 (B, n_bins, L)  
        logits = logits.permute(0, 2, 1)        # 因为 CrossEntropyLoss 要求类别维度在中间，这里把维度换个顺序方便算 loss

        # 只计算被 mask 的 bins 的 loss
        if mask.size(1) != logits.size(2):
            min_len = min(mask.size(1), logits.size(2))
            mask = mask[:, :min_len]
            logits = logits[:, :, :min_len]
            labels = labels[:, :min_len]

        masked_logits = logits[mask.unsqueeze(1).expand_as(logits)]
        masked_labels = labels[mask]
        if masked_logits.numel() == 0:
            return None  # 可能 batch 全未 mask

        loss = self.loss_fn(
            masked_logits.view(-1, self.n_bins),
            masked_labels.view(-1)
        )
        # 把 masked_logits 展平成 (N_masked, n_bins)
        # masked_labels 展平成 (N_masked,)
        # 计算交叉熵损失，看看模型预测的m/z等级对不对
        self.log_dict(
            {"train_loss": loss, "mask_rate": mask_rate},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    # Validation step
    def validation_step(self, batch, _):
        mzs, intensities = batch["mz_array"], batch["intensity_array"]
        #mask_rate = self.current_mask_rate()
        mask_rate = 0.05

        labels = self.bin_mz(mzs)
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate) # mask
        logits = self(masked_mzs, intensities).permute(0, 2, 1)

        # 只取出被 mask 的位置的预测结果（masked_logits）和真实标签（masked_labels）
        masked_logits = logits[mask.unsqueeze(1).expand_as(logits)]
        masked_labels = labels[mask]
        if masked_logits.numel() == 0:
            return None

        loss = self.loss_fn(
            masked_logits.view(-1, self.n_bins),
            masked_labels.view(-1)
        )
        self.log_dict(
            {"val_loss": loss, "mask_rate": mask_rate},
            on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss

    # Epoch-end logging
    def on_train_epoch_end(self):
        #mask_rate = self.current_mask_rate()
        mask_rate = 0.05
        avg_loss = self.trainer.callback_metrics.get("train_loss", None)
        if avg_loss is not None:
            avg_loss = avg_loss.item()
            self.train_losses.append(avg_loss)
        
        val_loss = self.trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            val_loss = val_loss.item()
            self.val_losses.append(val_loss)
            
        self.print(f"Epoch {self.current_epoch}: mask_rate={mask_rate:.2f}, train_loss={avg_loss:.4f}")
        
        # Plot and save loss curves every epoch
        self.plot_loss_curves()
    
    def plot_loss_curves(self):
        """Plot and save training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if self.val_losses:  # Plot validation loss if available
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.title('Loss Curves During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        plt.savefig(os.path.join('logs', 'loss_curves.png'))
        plt.close()

    # Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
