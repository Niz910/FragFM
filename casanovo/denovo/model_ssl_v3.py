import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl #import lightning.pytorch as pl
import torch.nn.functional as F
from casanovo.denovo.transformers import SpectrumEncoder
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple
# 注意参数 本模型适用于蛋白质
# forward阶段print mz值看看dataloader是否norm
# 用fourier也试试？
# plot validation loss 保证在图中有正常loss和validation loss
# 学习dreams是怎么用位置信息 另外学习一下loss的时候是不是cls预测还是这么个方式

class SpectrumSSLv2(pl.LightningModule):
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
        # n_bins 不再手动指定，由 mz_max/bin_size 自动计算
        min_mask: float = 0.05,
        max_mask: float = 0.5,
        total_epochs: int = 100,
        use_cosine_mask_schedule: bool = False,
        mz_min: float = 50.0,
        mz_max: float = 2500.0,
        log_scale: bool = True,
        bin_size: float = 0.5,  # ✅ 新增：分箱宽度

        # ✅
        max_peaks: int = 1000,
        # Masking config
        mask_prob: float = 0.15,  # used as overall probability; actual split is 80/10/10
        # Logging
        n_log: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 初始化参数
        self.lr = lr
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.bin_size = bin_size
        self.log_scale = log_scale
        self.MASK_TOKEN = -1.0 # # use -1.0 to distinguish from padding zeros

        ## ✅  Data + masking
        self.max_peaks = max_peaks
        self.mask_prob = mask_prob
        # Masking schedule (from model_ssl_liu)
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.total_epochs = total_epochs
        self.use_cosine_mask_schedule = use_cosine_mask_schedule


        # ✅ Auto-compute number of bins for DreaMS-style binning utility
        self.n_bins = int(math.ceil((self.mz_max - self.mz_min) / self.bin_size))
        print(f"[Init] n_bins automatically set to {self.n_bins}")

        # 日志记录
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Encoder
        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

        # ✅ MLP head for m/z regression (following model_ssl_v1.py structure)
        self.mz_predictor = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, 1),  # Output 1 value for continuous m/z prediction
        )
        self.mse_loss = nn.MSELoss()


        # Classification head – depends on n_bins
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, self.n_bins),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # ✅ Logging/plotting
        self.n_log = n_log
        self._history = []
        # Per-epoch aggregated losses for plotting
        self._plot_train_losses = []  # MSE (kept for backward compat)
        self._plot_val_losses = []    # MSE (kept for backward compat)
        self._plot_train_mse_losses = []
        self._plot_val_mse_losses = []
        self._plot_train_ce_losses = []
        self._plot_val_ce_losses = []
        self._plot_train_total_losses = []
        self._plot_val_total_losses = []

    # (mask rate schedule removed per request; using fixed mask_prob)

    # ----------------------------------------------------------------------
    # Mask 速率调度，保留备用
    # ----------------------------------------------------------------------
    def current_mask_rate(self):
        epoch = self.current_epoch
        min_mask = self.hparams.min_mask
        max_mask = self.hparams.max_mask
        total_epochs = max(1, self.hparams.total_epochs)

        if self.hparams.use_cosine_mask_schedule:
            rate = max_mask - (max_mask - min_mask) * 0.5 * (
                1 + math.cos(math.pi * epoch / total_epochs)
            )
        else:
            rate = min_mask + (max_mask - min_mask) * (epoch / total_epochs)
        return float(min(rate, max_mask))

    # ------------------------------------------------------------------
    # Fixed-width binning (refer to DreaMS)
    # ------------------------------------------------------------------
    def bin_mz(self, mzs: torch.Tensor) -> torch.Tensor:
        """
        DreaMS-style fixed-width binning (left-closed, right-open intervals):
        - Uses floor for binning
        - Clamps m/z to [mz_min, mz_max]
        - Bins limited to [0, n_bins-1]
        - Invalid (padded) positions (m/z <= 0) set to 0
        """
        valid_mask = mzs > 0
        if not valid_mask.any():
            return torch.zeros_like(mzs, dtype=torch.long)

        # 计算 floor 分箱
        clamped = torch.clamp(mzs, min=self.mz_min, max=self.mz_max)
        bins = torch.floor((clamped - self.mz_min) / self.bin_size).long()

        # 限制合法范围
        bins = bins.clamp(min=0, max=self.n_bins - 1)

        # 无效位置置零
        bins[~valid_mask] = 0
        return bins.detach()


    # ------------------------------------------------------------------
    # Masking strategy：Bert
    # ------------------------------------------------------------------
    def mask_spectrum(self, mzs: torch.Tensor, intensities: torch.Tensor, mask_rate: float):
        """
        借鉴selfsupervise.py的掩码策略，对m/z值进行掩码
        随机掩盖一部分峰的m/z值（mask）
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

    #  ✅ 先保留备用，提取了precursor信息
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a SpectrumDataset batch to tensors (mzs, intensities, precursors).
        Matches casanovo_model_BERT_maskold._process_batch.
        """
        precursor_mzs = batch["precursor_mz"].squeeze(0)
        precursor_charges = batch["precursor_charge"].squeeze(0)
        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
        precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T
        mzs = batch["mz_array"]
        intensities = batch["intensity_array"]
        return mzs, intensities, precursors

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, mzs: torch.Tensor, intensities: torch.Tensor):
        """
        Returns both regression (predicted m/z) and classification logits, plus padding mask.
        Outputs:
        - predicted_mz: (B, L-1) continuous m/z predictions
        - logits: (B, L-1, n_bins) classification logits
        - padding_mask: (B, L) boolean mask before dropping the first timestep
        """
        # Use padding mask like model_ssl_liu.py and drop the first timestep
        padding_mask = (mzs == 0)

        # ✅ 注意这里传入 encoder，告诉它哪些位置要屏蔽
        # ✅ liu's variable ‘enc’ change to ‘memories’，add regression head
        memories, _ = self.encoder(mzs, intensities, src_key_padding_mask=padding_mask)
    # memories = memories[:, 1:, :]  # 不再跳过第一个位置
        predicted_mz = self.mz_predictor(memories).squeeze(-1)  # (B, L-1)
        logits = self.mlp_head(memories)  # (B, L-1, n_bins)
        return predicted_mz, logits, padding_mask

    # ------------------------------------------------------------------
    # Train/Val steps (regression)
    # ------------------------------------------------------------------

    # ✅这里有些差异，加上了回归，有更复杂的处理和日志记录，
    def training_step(self, batch: Dict[str, torch.Tensor], *args, mode: str = "train") -> torch.Tensor:
        # ✅mzs, intensities = batch["mz_array"], batch["intensity_array"]
        mzs, intensities, precursors = self._process_batch(batch)

        # ✅Use fixed mask_prob as mask rate，liu=0.05
        mask_rate = self.mask_prob

        # Every 100 steps, print m/z stats to verify normalization/range
        try:
            if self.global_step % 100 == 0:
                valid = mzs > 0
                if valid.any():
                    vals = mzs[valid]
                    vmin = vals.min().detach().item()
                    vmax = vals.max().detach().item()
                    vmean = vals.mean().detach().item()
                    sample = vals.flatten()[:10].detach().cpu().numpy()
                    print(f"[v3] step {self.global_step}: m/z min={vmin:.2f} max={vmax:.2f} mean={vmean:.2f} | sample={np.round(sample,2).tolist()}")
        except Exception:
            pass

        # Sort by m/z ascending (match model_ssl_liu.py behavior)
        order = torch.argsort(mzs, dim=1)
        mzs = torch.gather(mzs, 1, order)
        intensities = torch.gather(intensities, 1, order)

        # Compute labels before masking
        labels = self.bin_mz(mzs)  # (B, L)
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate)

        # Build padding mask from original mzs (so masked tokens aren't treated as padding)
        padding_mask = (mzs == 0)
        memories, _ = self.encoder(masked_mzs, intensities, src_key_padding_mask=padding_mask)
    # memories = memories[:, 1:, :]  # 不再跳过第一个位置
        # Regression head
        predicted_mz = self.mz_predictor(memories).squeeze(-1)
        # Classification head
        logits = self.mlp_head(memories)  # (B, L, n_bins)

        # Align length if mismatch occurs (drop first timestep on targets/masks to match encoder output)
        mzs = mzs[:, 1:]
        intensities = intensities[:, 1:]
        labels = labels[:, 1:]
        mask = mask[:, 1:]
        padding_mask = padding_mask[:, 1:]
        min_len = min(predicted_mz.size(1), mzs.size(1))
        if predicted_mz.size(1) != mzs.size(1):
            predicted_mz = predicted_mz[:, :min_len]
            mzs = mzs[:, :min_len]
            intensities = intensities[:, :min_len]
            labels = labels[:, :min_len]
            mask = mask[:, :min_len]
            padding_mask = padding_mask[:, :min_len]
            logits = logits[:, :min_len, :]

        # Train on masked positions that are not padding
        train_mask = mask & (~padding_mask)
        # Early return if nothing to train on
        if train_mask.sum() == 0:
            zero = (predicted_mz.sum() + logits.sum()) * 0.0
            self.log(f"{mode}_Mask_Ratio", mask.float().mean().detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log(f"{mode}_mask_rate", torch.tensor(mask_rate, device=mzs.device), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            if mode == "train":
                self.log("train_regression_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
                self.log("train_classification_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
                self.log("train_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            return zero

        # Regression loss (MSE) on masked positions
        # ✅ Normalize MSE loss by m/z range to make it comparable to CE loss
        mz_range = self.mz_max - self.mz_min
        regression_loss = self.mse_loss(predicted_mz[train_mask], mzs[train_mask]) / (mz_range ** 2)
        # Classification loss (CE) on masked positions
        masked_logits = logits[train_mask]  # (N_mask, n_bins)
        masked_labels = labels[train_mask]  # (N_mask,)
        # Safety check for label range
        assert masked_labels.max() < self.n_bins, \
            f"label out of range: {masked_labels.max().item()} vs n_bins={self.n_bins}"

        classification_loss = self.loss_fn(masked_logits, masked_labels)
        # Total loss = regression + classification

        total_loss = regression_loss + classification_loss

        # Log MSE and RMSE for interpretability
        self.log(f"{mode}_MSE_Loss", regression_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_CE_Loss", classification_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Total_Loss", total_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=mzs.shape[0], sync_dist=True)
        try:
            # ✅ Log raw RMSE in m/z units (Th) for interpretability
            rmse_mz = torch.sqrt(self.mse_loss(predicted_mz[train_mask], mzs[train_mask]))
            self.log(f"{mode}_RMSE_mz", rmse_mz.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        except Exception:
            pass

        if mode == "train":
            self.log("train_regression_loss", regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("train_classification_loss", classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("train_loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        # Masking stats
        self.log(f"{mode}_Mask_Ratio", mask.float().mean().detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # Also log the effective mask_rate used this epoch
        self.log(f"{mode}_mask_rate", torch.tensor(mask_rate, device=mzs.device), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        return total_loss

    # ✅ 同样有些差异
    def validation_step(self, batch: Dict[str, torch.Tensor], *args) -> torch.Tensor:
        mzs, intensities, precursors = self._process_batch(batch)
        # Sort by m/z ascending to match training behavior

        order = torch.argsort(mzs, dim=1)
        mzs = torch.gather(mzs, 1, order)
        intensities = torch.gather(intensities, 1, order)
        # Use fixed mask_prob as mask rate

        mask_rate = self.mask_prob
        # Compute labels before masking
        labels = self.bin_mz(mzs)
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate)
        padding_mask = (mzs == 0)
        memories, _ = self.encoder(masked_mzs, intensities, src_key_padding_mask=padding_mask)
    # memories = memories[:, 1:, :]  # 不再跳过第一个位置

        # Regression head
        predicted_mz = self.mz_predictor(memories).squeeze(-1)
        # Classification head
        logits = self.mlp_head(memories)  # (B, L, n_bins)

        # Align length similarly to training (drop first timestep and trim equally)
        mzs = mzs[:, 1:]
        intensities = intensities[:, 1:]
        labels = labels[:, 1:]
        mask = mask[:, 1:]
        padding_mask = padding_mask[:, 1:]
        min_len = min(predicted_mz.size(1), mzs.size(1))
        if predicted_mz.size(1) != mzs.size(1):
            predicted_mz = predicted_mz[:, :min_len]
            mzs = mzs[:, :min_len]
            intensities = intensities[:, :min_len]
            labels = labels[:, :min_len]
            mask = mask[:, :min_len]
            padding_mask = padding_mask[:, :min_len]
            logits = logits[:, :min_len, :]

        # Only masked and non-padding positions
        val_mask = mask & (~padding_mask)
        if val_mask.sum() == 0:
            zero = (predicted_mz.sum() + logits.sum()) * 0.0
            self.log("valid_MSE_Loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("valid_CE_Loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("val_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            return zero

        # ✅ Normalize MSE loss by m/z range (same as training)
        mz_range = self.mz_max - self.mz_min
        regression_loss = self.mse_loss(predicted_mz[val_mask], mzs[val_mask]) / (mz_range ** 2)
        masked_logits = logits[val_mask]
        masked_labels = labels[val_mask]
        assert masked_labels.max() < self.n_bins, f"label out of range: {masked_labels.max().item()} vs n_bins={self.n_bins}"
        classification_loss = self.loss_fn(masked_logits, masked_labels)
        total_loss = regression_loss + classification_loss

        self.log("valid_MSE_Loss", regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_CE_Loss", classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # Log total loss for validation explicitly for plotting
        self.log("valid_Total_Loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        try:
            # ✅ Log raw RMSE in m/z units (Th) for interpretability
            rmse_mz = torch.sqrt(self.mse_loss(predicted_mz[val_mask], mzs[val_mask]))
            self.log("valid_RMSE_mz", rmse_mz.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        except Exception:
            pass
        self.log("val_loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        return total_loss

    # ------------------------------------------------------------------
    # Epoch end hooks and plotting (adapted)
    # ------------------------------------------------------------------
    # ✅ plot略有差异
    def on_train_epoch_end(self) -> None:
        callback_metrics = self.trainer.callback_metrics
        train_mse = callback_metrics.get("train_MSE_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        # Try to read CE/Total aggregated metrics if they exist
        try:
            train_ce = callback_metrics.get("train_CE_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        except Exception:
            train_ce = float("nan")
        try:
            train_total = callback_metrics.get("train_Total_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        except Exception:
            train_total = float("nan")

        # Validation epoch metrics (Lightning exposes them in callback_metrics)
        valid_mse = callback_metrics.get("valid_MSE_Loss", torch.tensor(float("nan"))).detach().item()
        try:
            valid_ce = callback_metrics.get("valid_CE_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_ce = float("nan")
        try:
            valid_total = callback_metrics.get("valid_Total_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_total = float("nan")

        mask_ratio = callback_metrics.get("train_Mask_Ratio", torch.tensor(float("nan"))).detach().item()

        # Unified epoch summary printing (train + valid, mse/ce/total)
        print(f"\nEpoch {self.current_epoch} Summary (SSL v3):")
        print(f"  Train MSE:   {train_mse:.6f}")
        print(f"  Train CE:    {train_ce:.6f}")
        print(f"  Train Total: {train_total:.6f}")
        print(f"  Valid MSE:   {valid_mse:.6f}")
        print(f"  Valid CE:    {valid_ce:.6f}")
        print(f"  Valid Total: {valid_total:.6f}")
        print(f"  Mask Ratio:  {mask_ratio:.2%}")

        metrics = {
            "step": self.trainer.global_step,
            "train_mse": train_mse,
            "mask_ratio": mask_ratio,
        }
        self._history.append(metrics)
        # Collect per-epoch aggregated metrics for plotting
        self._plot_train_losses.append(metrics.get("train_mse", float("nan")))  # legacy (MSE)
        self._plot_train_mse_losses.append(metrics.get("train_mse", float("nan")))
        self._plot_train_ce_losses.append(train_ce)
        # train Total loss if available
        try:
            train_total = self.trainer.callback_metrics.get("train_Total_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        except Exception:
            train_total = float("nan")
        self._plot_train_total_losses.append(train_total)
        self._log_history()
        self._plot_loss_curves()

    def _plot_loss_curves(self) -> None:
        try:
            os.makedirs('logs', exist_ok=True)

            # MSE curves
            if len(self._plot_train_mse_losses) > 0 or len(self._plot_val_mse_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_mse_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_mse_losses) + 1), self._plot_train_mse_losses, 'b-', label='Train MSE', linewidth=2)
                if len(self._plot_val_mse_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_mse_losses) + 1), self._plot_val_mse_losses, 'r-', label='Valid MSE', linewidth=2)
                plt.title('SSL v3: MSE Loss')
                plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v3_mse.png'))
                plt.close()

            # CE curves
            if len(self._plot_train_ce_losses) > 0 or len(self._plot_val_ce_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_ce_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_ce_losses) + 1), self._plot_train_ce_losses, 'b-', label='Train CE', linewidth=2)
                if len(self._plot_val_ce_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_ce_losses) + 1), self._plot_val_ce_losses, 'r-', label='Valid CE', linewidth=2)
                plt.title('SSL v3: Cross-Entropy Loss')
                plt.xlabel('Epoch'); plt.ylabel('CE Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v3_ce.png'))
                plt.close()

            # Total loss curves
            if len(self._plot_train_total_losses) > 0 or len(self._plot_val_total_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_total_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_total_losses) + 1), self._plot_train_total_losses, 'b-', label='Train Total', linewidth=2)
                if len(self._plot_val_total_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_total_losses) + 1), self._plot_val_total_losses, 'r-', label='Valid Total', linewidth=2)
                plt.title('SSL v3: Total Loss (MSE + CE)')
                plt.xlabel('Epoch'); plt.ylabel('Total Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v3_total.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # 创建学习率调度器
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.total_epochs,
                pct_start=0.3,  # 预热阶段占总步数的30%
                cycle_momentum=False
            ),
            'interval': 'epoch',  # 每个epoch更新一次
            'name': 'learning_rate'
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


    # ✅ 增加这个，先留着吧
    def on_validation_epoch_end(self) -> None:
        callback_metrics = self.trainer.callback_metrics
        valid_mse = callback_metrics.get("valid_MSE_Loss", torch.tensor(float("nan"))).detach().item()
        # Collect CE if present
        try:
            valid_ce = callback_metrics.get("valid_CE_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_ce = float("nan")
        # Collect Total if present
        try:
            valid_total = callback_metrics.get("valid_Total_Loss", torch.tensor(float("nan"))).detach().item()
        except Exception:
            valid_total = float("nan")
        metrics = {"step": self.trainer.global_step, "valid_mse": valid_mse}
        self._history.append(metrics)
        self._plot_val_losses.append(metrics.get("valid_mse", float("nan")))  # legacy (MSE)
        self._plot_val_mse_losses.append(metrics.get("valid_mse", float("nan")))
        self._plot_val_ce_losses.append(valid_ce)
        self._plot_val_total_losses.append(valid_total)
        self._log_history()
        self._plot_loss_curves()

    # ✅ 更复杂的日志记录，先留着吧
    def _log_history(self) -> None:
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            print("Step\tTrain MSE\tValid MSE\tMask%")
        metrics = self._history[-1]
        if metrics.get("step", 0) % self.n_log == 0:
            vals = [
                metrics.get("step", -1),
                metrics.get("train_mse", float("nan")),
                metrics.get("valid_mse", float("nan")),
                (metrics.get("mask_ratio", float("nan")) or 0.0) * 100,
            ]
            print("%i\t%.6f\t%.6f\t%.2f" % tuple(vals))


# Backward-compatible alias if desired
SpectrumSSL = SpectrumSSLv2

