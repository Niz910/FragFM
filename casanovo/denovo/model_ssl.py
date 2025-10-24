import math
import torch
import torch.nn as nn
import pytorch_lightning as pl #import lightning.pytorch as pl
import torch.nn.functional as F
from casanovo.denovo.transformers import SpectrumEncoder
# 改预测目标 改成m/z 删掉余弦 
# 读DreaMS/casanovo encoder
class SpectrumSSL(pl.LightningModule):
    """
    Self-supervised spectrum modeling (encoder-only).
    Curriculum masking + binning + cross-entropy loss.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        lr: float = 1e-4,
        n_bins: int = 64,
        min_mask: float = 0.05,
        max_mask: float = 0.5,
        total_epochs: int = 100,
        use_cosine_mask_schedule: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

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
            nn.Linear(dim_model, n_bins),
        )

        self.lr = lr
        self.n_bins = n_bins
        self.loss_fn = nn.CrossEntropyLoss()# 这里有softmax层所以训练阶段就不用了

    # Mask rate scheduler
    def current_mask_rate(self):
        return 0.15

    # 把m/z改成分类bin变量
    def bin_mz(self, mzs: torch.Tensor) -> torch.Tensor:
        """
        把 m/z 映射到 [0, n_bins-1] 的整数类别。
        """
        max_per_spec = mzs.max(dim=1, keepdim=True)[0]
        max_per_spec[max_per_spec == 0] = 1e-8
        normed = mzs / max_per_spec
        bins = torch.clamp(
            (normed * (self.n_bins - 1)).long(),
            min=0, max=self.n_bins - 1
        )
        return bins.detach()

    # 根据 m/z 与 intensity（用于跳过 padding）生成 mask 后的 m/z
    def mask_mz(self, mzs: torch.Tensor, intensities: torch.Tensor, mask_rate: float):
        B, L = mzs.shape
        device = mzs.device
        # 有效位置：非 padding
        valid = (intensities > 0) & (mzs > 0)
        rand = torch.rand(B, L, device=device)
        mask = (rand < mask_rate) & valid

        masked_mz = mzs.clone()
        masked_mz[mask] = 0.0
        return masked_mz, mask

    # Forward encoder➕MLP
    def forward(self, mzs, intensities):
        enc, _ = self.encoder(mzs, intensities)
        enc = enc[:, 1:, :]  # 🩹 去掉 CLS token（SpectrumEncoder 自动加的）
        logits = self.mlp_head(enc)  # (B, L, n_bins)
        return logits

    # Training step
    def training_step(self, batch, _):
        mzs = batch["mz_array"]
        intensities = batch["intensity_array"]
        mask_rate = self.current_mask_rate()

        labels = self.bin_mz(mzs)                                   # ▶️ m/z 作为标签
        masked_mz, mask = self.mask_mz(mzs, intensities, mask_rate) # ▶️ 掩码 m/z

        logits = self(masked_mz, intensities)                        # ▶️ 用 masked_mz 做输入
        # logits: (B, L, n_bins)
        masked_logits = logits[mask]                                 # (N_masked, n_bins)
        masked_labels = labels[mask]                                 # (N_masked,)

        if masked_logits.numel() == 0:
            return None

        loss = self.loss_fn(masked_logits, masked_labels)
        self.log_dict({"train_loss": loss, "mask_rate": mask_rate},
                    on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # Validation step
    def validation_step(self, batch, _):
        mzs = batch["mz_array"]
        intensities = batch["intensity_array"]
        mask_rate = self.current_mask_rate()

        labels = self.bin_mz(mzs)
        masked_mz, mask = self.mask_mz(mzs, intensities, mask_rate)

        logits = self(masked_mz, intensities)
        masked_logits = logits[mask]
        masked_labels = labels[mask]

        if masked_logits.numel() == 0:
            return None

        loss = self.loss_fn(masked_logits, masked_labels)
        self.log_dict({"val_loss": loss, "mask_rate": mask_rate},
                    on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # Epoch-end logging
    def on_train_epoch_end(self):
        mask_rate = self.current_mask_rate()
        avg_loss = self.trainer.callback_metrics.get("train_loss", None)
        if avg_loss is not None:
            avg_loss = avg_loss.item()
        self.print(
            f"[Epoch {self.current_epoch}] mask_rate={mask_rate:.2f}, train_loss={avg_loss:.4f} → predicting masked m/z bins"
        )

    # Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
