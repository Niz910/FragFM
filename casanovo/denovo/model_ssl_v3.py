import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl #import lightning.pytorch as pl
import torch.nn.functional as F
from casanovo.denovo.transformers import SpectrumEncoder
from depthcharge.encoders import FloatEncoder
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
        self.MASK_TOKEN = -1.0 # # use -1.0 to distinguish from padding zeros

        ## ✅  Data + masking
        self.max_peaks = max_peaks
        self.mask_prob = mask_prob
        # Masking schedule (from model_ssl_liu)
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.total_epochs = total_epochs


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

        # ✅ Precursor encoders：分别编码charge、mass、mz
        # charge使用embedding (charge范围通常是1-10的整数)
        max_charge = 10  # 假设最大charge为10
        self.charge_encoder = nn.Embedding(max_charge + 1, dim_model)  # +1是为了包含0-charge
        
        # mass和mz使用FloatEncoder
        self.mass_encoder = FloatEncoder(d_model=dim_model)
        self.mz_encoder = FloatEncoder(d_model=dim_model)

        # ✅ MLP head for m/z regression (following model_ssl_v1.py structure)
        self.mz_predictor = nn.Sequential(
            nn.Linear(dim_model, dim_model // 4),  # 使用整数除法
            nn.ReLU(),
            nn.Linear(dim_model // 4, 1),  # 修正第二层的输入维度
        )
        self.mse_loss = nn.MSELoss()


        # Classification head – depends on n_bins
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_model, dim_model // 4),
            nn.ReLU(),
            nn.Linear(dim_model // 4, self.n_bins),
        )
        # === Cross-Attention module: precursor attends to [CLS, peaks] ===
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=n_head,
            batch_first=True  # 让输入输出保持 (B, L, D) 格式
        )
        self.cross_attn_ln = torch.nn.LayerNorm(dim_model)
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

    # ------------------------------------------------------------------
    # Fixed-width binning (refer to DreaMS)
    # ------------------------------------------------------------------
    def bin_mz(self, mzs: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        """
        DreaMS-style fixed-width binning (left-closed, right-open intervals):
        - Uses floor for binning
        - Clamps m/z to [mz_min, mz_max] or [0, 1] if normalized
        - Bins limited to [0, n_bins-1]
        - Invalid (padded) positions (m/z <= 0) set to 0
        
        Args:
            mzs: m/z values tensor
            is_normalized: True if mzs are already normalized to [0,1], False if in original scale
        """
        valid_mask = mzs > 0
        if not valid_mask.any():
            return torch.zeros_like(mzs, dtype=torch.long)

        if is_normalized:
            # ✅ 处理归一化的m/z值 [0,1] -> bins
            # 归一化值直接乘以n_bins然后floor
            clamped = torch.clamp(mzs, min=0.0, max=1.0)
            bins = torch.floor(clamped * self.n_bins).long()
        else:
            # 原始逻辑：处理原始m/z值
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
        对 m/z 进行 BERT 风格掩码：
        - 80% 替换为 MASK_TOKEN
        - 10% 替换为当前光谱的真实峰（随机）
        - 10% 保持原值
        """
        B, L = mzs.shape

        # 只对真实峰（>0）进行掩码
        valid_mask = mzs > 0
        mask = torch.zeros_like(mzs, dtype=torch.bool)

        # 决定哪些位置要 mask
        for b in range(B):
            valid_indices = torch.where(valid_mask[b])[0]
            if len(valid_indices) > 0:
                n_mask = max(1, int(mask_rate * len(valid_indices)))
                chosen_indices = valid_indices[torch.randperm(len(valid_indices))[:n_mask]]
                mask[b, chosen_indices] = True

        masked_mzs = mzs.clone()

        # 对被 mask 的位置做 80/10/10 替换
        for b in range(B):
            masked_positions = torch.where(mask[b])[0]

            # 🟢 正确：每个样本都重新计算自己的有效峰
            valid_indices = torch.where(valid_mask[b])[0]
            valid_values = mzs[b, valid_indices]

            for pos in masked_positions:
                r = torch.rand(1).item()
                if r < 0.8:
                    masked_mzs[b, pos] = self.MASK_TOKEN
                elif r < 0.9:
                    # 10% → 随机真实峰（严格来自当前 b）
                    rand_idx = torch.randint(0, len(valid_values), (1,), device=mzs.device)
                    masked_mzs[b, pos] = valid_values[rand_idx]
                # elif r < 0.9:  # 10%概率：用随机值替换
                #     # ✅ 更新：为归一化的m/z生成[0,1]范围内的随机值
                #     masked_mzs[b, pos] = torch.rand(1).item()  # 直接生成[0,1]范围的随机值
                # # else: 10%概率保持原值不变
                else:
                    # 10% → 保持原值，不改
                    pass

        return masked_mzs, mask

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
                elif r < 0.9:  # 10%: replace with random *real* m/z value new*
                    valid_values = mzs[b, valid_indices]            # 所有真实峰
                    rand_idx = torch.randint(0, len(valid_values), (1,), device=mzs.device)
                    masked_mzs[b, pos] = valid_values[rand_idx]     # 取真实峰
                
                # elif r < 0.9:  # 10%概率：用随机值替换
                #     # ✅ 更新：为归一化的m/z生成[0,1]范围内的随机值
                #     masked_mzs[b, pos] = torch.rand(1).item()  # 直接生成[0,1]范围的随机值
                # # else: 10%概率保持原值不变
        
        return masked_mzs, mask

    #  ✅ 修改为返回完整的precursor信息
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a SpectrumDataset batch to tensors (mzs, intensities, precursors).
        ✅ 返回完整的precursor信息：[mass, charge, mz]
        """
        precursor_mzs = batch["precursor_mz"].squeeze(0)  # Shape: (batch_size,)
        precursor_charges = batch["precursor_charge"].squeeze(0)  # Shape: (batch_size,)
        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges  # Shape: (batch_size,)
        
        # 组合成完整的precursor信息：[mass, charge, mz]
        precursors = torch.stack([precursor_masses, precursor_charges, precursor_mzs], dim=1)  # Shape: (batch_size, 3)
        
        mzs = batch["mz_array"]  # Shape: (batch_size, max_peaks)
        intensities = batch["intensity_array"]  # Shape: (batch_size, max_peaks)
        
        # ✅ 新方案：在数据处理阶段就进行m/z归一化
        # 将m/z值归一化到[0,1]范围，避免loss计算时的梯度压缩问题
        mz_range = self.mz_max - self.mz_min  # 2450
        
        # 只对有效的m/z值进行归一化（padding位置保持为0）
        valid_mask = mzs > 0
        normalized_mzs = mzs.clone()
        normalized_mzs[valid_mask] = (mzs[valid_mask] - self.mz_min) / mz_range
        
        # 对precursor的mz也进行相同的归一化
        normalized_precursors = precursors.clone()
        normalized_precursors[:, 2] = (precursors[:, 2] - self.mz_min) / mz_range  # 归一化precursor mz
        
        return normalized_mzs, intensities, normalized_precursors

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, mzs: torch.Tensor, intensities: torch.Tensor, precursors: torch.Tensor):
        """
        ✅ 更新为包含完整precursor信息的forward函数
        序列结构：1+1+150=152：cls+precursor+序列
        
        Args:
            mzs: (B, L) m/z values for peaks
            intensities: (B, L) intensity values for peaks
            precursors: (B, 3) precursor信息 [mass, charge, mz]
            
        Returns:
            - predicted_mz: (B, L) continuous m/z predictions for all peaks
            - logits: (B, L, n_bins) classification logits for all peaks
            - padding_mask: (B, L+2) boolean mask for the full sequence (cls+precursor+peaks)
            - full_features: (B, L+2, d_model) embedding features for the full sequence
        """
        B, L = mzs.shape
        
        # 1. 分别编码precursor的三个组成部分：mass, charge, mz
        # precursors: (B, 3) -> [mass, charge, mz]
        masses = precursors[:, 0]      # (B,) - mass values
        charges = precursors[:, 1]     # (B,) - charge values  
        mz_values = precursors[:, 2]   # (B,) - m/z values
        
        # 编码各个组件
        mass_encoded = self.mass_encoder(masses.unsqueeze(-1)).squeeze(1)  # (B, d_model)
        charge_encoded = self.charge_encoder(charges.long())  # (B, d_model) - charge转为整数索引
        mz_encoded = self.mz_encoder(mz_values.unsqueeze(-1)).squeeze(1)  # (B, d_model)
        
        # 将三个编码相加得到最终的precursor表示
        precursors_combined = mass_encoded + charge_encoded + mz_encoded  # (B, d_model)
        precursor_token = precursors_combined.unsqueeze(1)  # (B, 1, d_model)
        
        """
        第一部分是在“造特征”——先把 [CLS+peaks] 用 self-attention 编码成 spectrum_memories，再用 cross-attention
        让 precursor_token 从 memory=[CLS+peaks] 里取信息并被“条件化”（得到 precursor_token 的 refined 版本），
        最后把三者拼成 full_sequence = [CLS, precursor, peaks]，供后续预测用。
        """
        # 2. 原始的spectrum encoding (不包括precursor)
        spectrum_padding_mask = (mzs == 0)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mzs.device)
        #new* 把第一个token看作false，就是不把cls设成padding，只参与self attention但不进行预测
        spectrum_padding_mask = torch.cat([cls_mask, spectrum_padding_mask], dim=1) 
        spectrum_memories, _ = self.encoder(mzs, intensities, src_key_padding_mask=spectrum_padding_mask)  # (B, L, d_model)
        memory = spectrum_memories

        # === (2.5) Cross-Attention：Q=precursor, K/V=[CLS, peaks] === *new
        spectrum_q = memory               # (B, L+1, D)
        precursor_kv = precursor_token    # (B, 1, D)

        spectrum_refined, attn_weights = self.cross_attn(
            query=spectrum_q,
            key=precursor_kv,
            value=precursor_kv,
            key_padding_mask=None,   # precursor 只有一个 token，不需要 mask
        )
        memory = self.cross_attn_ln(memory + spectrum_refined)   # 残差 + LN 假发残差保证稳定性用的
        # 3. 拼接序列：[CLS] + [PRECURSOR] + [PEAKS...]
        # CLS token 已经在 SpectrumEncoder 内部处理
        # 我们需要在CLS和peaks之间插入precursor
        # spectrum_memories 的第一个位置是CLS，后面是peaks
        cls_token = memory[:, 0:1, :]  # (B, 1, d_model) - CLS token
        peak_tokens = memory[:, 1:, :]  # (B, L, d_model) - Peak tokens
        
        # 拼接：CLS + PRECURSOR + PEAKS
        full_sequence = torch.cat([cls_token, precursor_token, peak_tokens], dim=1)  # (B, 1+1+L, d_model) = (B, L+2, d_model)
        
        """
        第二部分是在“出结果 + 做掩码”——给拼好的新序列配套一个完整的 padding mask，然后把 full_sequence 
        喂进预测头，产出所有位置的预测，再只抽取 peaks 部分用于损失/评估。
        """
        # 4. 生成完整的padding mask：[CLS] + [PRECURSOR] + [PEAKS...]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mzs.device)  # CLS不是padding
        precursor_mask = torch.zeros(B, 1, dtype=torch.bool, device=mzs.device)  # PRECURSOR不是padding
        peak_mask = spectrum_padding_mask[:, 1:] # new*跳过mask掉的cls
        full_padding_mask = torch.cat([cls_mask, precursor_mask, peak_mask], dim=1)  # (B, L+2)
        
        # 5. 预测头：使用完整序列信息（CLS + PRECURSOR + PEAKS）进行预测
        # 让每个peak都能看到CLS和PRECURSOR的信息，提高预测准确性
        full_features = full_sequence  # (B, L+2, d_model) - 完整序列：CLS + PRECURSOR + PEAKS
        # 开始decoder
        all_predicted_mz = self.mz_predictor(full_features).squeeze(-1)  # (B, L+2)
        all_logits = self.mlp_head(full_features)  # (B, L+2, n_bins)
        
        # 只取peaks部分的预测结果用于loss计算
        predicted_mz = all_predicted_mz[:, 2:]  # (B, L) - 只要peaks的预测
        logits = all_logits[:, 2:, :]  # (B, L, n_bins) - 只要peaks的预测
        
        return predicted_mz, logits, full_padding_mask, full_features

    # ------------------------------------------------------------------
    # Train/Val steps (regression)
    # ------------------------------------------------------------------

    # ✅更新为包含precursor信息的训练步骤
    def training_step(self, batch: Dict[str, torch.Tensor], *args, mode: str = "train") -> torch.Tensor:
        # ✅获取mzs, intensities, precursors
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
                    print(f"[v4] step {self.global_step}: m/z min={vmin:.2f} max={vmax:.2f} mean={vmean:.2f} | sample={np.round(sample,2).tolist()}")
        except Exception:
            pass

        # Sort by m/z ascending (match model_ssl_liu.py behavior)
        order = torch.argsort(mzs, dim=1)
        mzs = torch.gather(mzs, 1, order)
        intensities = torch.gather(intensities, 1, order)

        # Compute labels before masking
        # ✅ 更新：由于mzs现在是归一化的，需要使用is_normalized=True
        labels = self.bin_mz(mzs, is_normalized=True)  # (B, L)
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate)

        # ✅ 使用新的forward函数，包含precursor信息
        predicted_mz, logits, full_padding_mask, full_features = self.forward(masked_mzs, intensities, precursors)
        # predicted_mz: (B, L), logits: (B, L, n_bins), full_padding_mask: (B, L+2), full_features: (B, L+2, d_model)

        # ✅ 调整padding_mask以匹配输出
        # 注意：SpectrumEncoder已经处理了CLS token，我们的predicted_mz对应原始的所有peaks
        # 不需要删除第一个元素，因为predicted_mz覆盖所有原始peaks
        padding_mask = full_padding_mask[:, 2:]  # (B, L) - 跳过CLS和PRECURSOR，只要peaks部分

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
        # ✅ 新方案：直接在归一化空间计算MSE，无需额外归一化
        # 因为m/z已经在[0,1]范围，MSE自然在[0,1]范围，与CE数量级相当
        raw_mse = self.mse_loss(predicted_mz[train_mask], mzs[train_mask])
        regression_loss = raw_mse  # 直接使用raw_mse，无需除法归一化
        
        # 计算原始m/z单位的RMSE用于监控（恢复到原始尺度）
        mz_range = self.mz_max - self.mz_min
        rmse_mz_units = torch.sqrt(raw_mse) * mz_range  # 恢复到Th单位
        
        # Classification loss (CE) on masked positions
        masked_logits = logits[train_mask]  # (N_mask, n_bins)
        masked_labels = labels[train_mask]  # (N_mask,)
        # Safety check for label range
        assert masked_labels.max() < self.n_bins, \
            f"label out of range: {masked_labels.max().item()} vs n_bins={self.n_bins}"

        classification_loss = self.loss_fn(masked_logits, masked_labels)
        
        # ✅ 动态权重策略：初期以分类为主，后期以回归为主
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
        
        # 计算权重：初期CE权重高，后期MSE权重高
        # epoch 0-30%: CE主导 (CE:1.0, MSE:0.3)
        # epoch 30%-70%: 平衡过渡 (CE:1.0->0.5, MSE:0.3->1.0) 
        # epoch 70%-100%: MSE主导 (CE:0.5, MSE:1.0)
        epoch_progress = current_epoch / total_epochs
        
        if epoch_progress < 0.3:
            # 初期：分类为主
            ce_weight = 1.0
            mse_weight = 0.3
        elif epoch_progress < 0.7:
            # 中期：线性过渡
            transition_progress = (epoch_progress - 0.3) / 0.4  # 0到1
            ce_weight = 1.0 - 0.5 * transition_progress  # 1.0 -> 0.5
            mse_weight = 0.3 + 0.7 * transition_progress  # 0.3 -> 1.0
        else:
            # 后期：回归为主
            ce_weight = 0.5
            mse_weight = 1.0
            
        # Total loss = 加权组合
        weighted_regression_loss = mse_weight * regression_loss
        weighted_classification_loss = ce_weight * classification_loss
        total_loss = weighted_regression_loss + weighted_classification_loss

        # Log MSE and RMSE for interpretability
        self.log(f"{mode}_MSE_Loss", regression_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_CE_Loss", classification_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Total_Loss", total_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # ✅ 记录权重和加权损失
        self.log(f"{mode}_CE_Weight", ce_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_MSE_Weight", mse_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Weighted_MSE_Loss", weighted_regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Weighted_CE_Loss", weighted_classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log(f"{mode}_Raw_MSE", raw_mse.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # ✅ 新方案：记录原始m/z单位的RMSE（恢复尺度后）
        self.log(f"{mode}_RMSE_mz_units", rmse_mz_units.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        if mode == "train":
            self.log("train_regression_loss", weighted_regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("train_classification_loss", weighted_classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("train_loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        # Masking stats
        self.log(f"{mode}_Mask_Ratio", mask.float().mean().detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # Also log the effective mask_rate used this epoch
        self.log(f"{mode}_mask_rate", torch.tensor(mask_rate, device=mzs.device), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)

        # return total_loss  # 原始代码：使用加权总损失
        return regression_loss  # ✅ 只用MSE进行反向传播，不使用CE loss

    # ✅ 更新为包含precursor信息的验证步骤
    def validation_step(self, batch: Dict[str, torch.Tensor], *args) -> torch.Tensor:
        mzs, intensities, precursors = self._process_batch(batch)
        # Sort by m/z ascending to match training behavior

        order = torch.argsort(mzs, dim=1)
        mzs = torch.gather(mzs, 1, order)
        intensities = torch.gather(intensities, 1, order)
        # Use fixed mask_prob as mask rate

        mask_rate = self.mask_prob
        # Compute labels before masking
        # ✅ 更新：由于mzs现在是归一化的，需要使用is_normalized=True
        labels = self.bin_mz(mzs, is_normalized=True)
        masked_mzs, mask = self.mask_spectrum(mzs, intensities, mask_rate)
        
        # ✅ 使用新的forward函数，包含precursor信息
        predicted_mz, logits, full_padding_mask, full_features = self.forward(masked_mzs, intensities, precursors)

        # ✅ 调整目标和掩码长度以匹配输出
        # 注意：SpectrumEncoder已经处理了CLS token，我们的predicted_mz对应原始的所有peaks
        # 不需要删除第一个元素，因为predicted_mz覆盖所有原始peaks
        # mzs, labels, mask保持原始形状以匹配predicted_mz
        # 只需要调整padding_mask从full_padding_mask中提取peaks部分
        padding_mask = full_padding_mask[:, 2:]  # (B, L) - 跳过CLS和PRECURSOR，只要peaks部分

        # Only masked and non-padding positions
        val_mask = mask & (~padding_mask)
        if val_mask.sum() == 0:
            zero = (predicted_mz.sum() + logits.sum()) * 0.0
            self.log("valid_MSE_Loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("valid_CE_Loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            self.log("val_loss", zero.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
            return zero

        # ✅ 使用新方案的loss计算 (与训练保持一致)
        raw_mse = self.mse_loss(predicted_mz[val_mask], mzs[val_mask])
        regression_loss = raw_mse  # 直接使用raw_mse，无需除法归一化
        
        # 计算原始m/z单位的RMSE用于监控
        mz_range = self.mz_max - self.mz_min
        rmse_mz_units = torch.sqrt(raw_mse) * mz_range
        
        masked_logits = logits[val_mask]
        masked_labels = labels[val_mask]
        assert masked_labels.max() < self.n_bins, f"label out of range: {masked_labels.max().item()} vs n_bins={self.n_bins}"
        classification_loss = self.loss_fn(masked_logits, masked_labels)
        
        # 相同的动态权重策略
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
        epoch_progress = current_epoch / total_epochs
        
        if epoch_progress < 0.3:
            ce_weight = 1.0
            mse_weight = 0.3
        elif epoch_progress < 0.7:
            transition_progress = (epoch_progress - 0.3) / 0.4
            ce_weight = 1.0 - 0.5 * transition_progress
            mse_weight = 0.3 + 0.7 * transition_progress
        else:
            ce_weight = 0.5
            mse_weight = 1.0
            
        weighted_regression_loss = mse_weight * regression_loss
        weighted_classification_loss = ce_weight * classification_loss
        total_loss = weighted_regression_loss + weighted_classification_loss

        self.log("valid_MSE_Loss", regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_CE_Loss", classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # Log total loss for validation explicitly for plotting
        self.log("valid_Total_Loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # ✅ 记录验证阶段的权重和加权损失
        self.log("valid_CE_Weight", ce_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_MSE_Weight", mse_weight, on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_Weighted_MSE_Loss", weighted_regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_Weighted_CE_Loss", weighted_classification_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        self.log("valid_Raw_MSE", raw_mse.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        
        # ✅ 新方案：记录原始m/z单位的RMSE（恢复尺度后）
        self.log("valid_RMSE_mz_units", rmse_mz_units.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)
        # self.log("val_loss", total_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)  # 原始代码
        self.log("val_loss", regression_loss.detach(), on_step=False, on_epoch=True, batch_size=mzs.shape[0], sync_dist=True)  # ✅ 改为regression_loss

        # return total_loss  # 原始代码：使用加权总损失
        return regression_loss  # ✅ 只用MSE进行反向传播，不使用CE loss

    # ------------------------------------------------------------------
    # Epoch end hooks and plotting (adapted)
    # ------------------------------------------------------------------
    # ✅ plot略有差异
    def on_train_epoch_end(self) -> None:
        callback_metrics = self.trainer.callback_metrics
        train_mse = callback_metrics.get("train_MSE_Loss_epoch", torch.tensor(float("nan"))).detach().item()
        
        # ✅ 保存最后一个epoch的模型为.pt文件
        if self.current_epoch == self.trainer.max_epochs - 1:
            pt_save_path = os.path.join(self.trainer.default_root_dir, f"model_ssl_v4_epoch{self.current_epoch:03d}.pt")
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.state_dict(),
                'hparams': self.hparams,
            }, pt_save_path)
            print(f"\n✅ Model saved to {pt_save_path}")
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
        print(f"\nEpoch {self.current_epoch} Summary (SSL v4):")
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
                plt.title('SSL v4: MSE Loss')
                plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v4_mse.png'))
                plt.close()

            # CE curves
            if len(self._plot_train_ce_losses) > 0 or len(self._plot_val_ce_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_ce_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_ce_losses) + 1), self._plot_train_ce_losses, 'b-', label='Train CE', linewidth=2)
                if len(self._plot_val_ce_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_ce_losses) + 1), self._plot_val_ce_losses, 'r-', label='Valid CE', linewidth=2)
                plt.title('SSL v4: Cross-Entropy Loss')
                plt.xlabel('Epoch'); plt.ylabel('CE Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v4_ce.png'))
                plt.close()

            # Total loss curves
            if len(self._plot_train_total_losses) > 0 or len(self._plot_val_total_losses) > 0:
                plt.figure(figsize=(10, 6))
                if len(self._plot_train_total_losses) > 0:
                    plt.plot(range(1, len(self._plot_train_total_losses) + 1), self._plot_train_total_losses, 'b-', label='Train Total', linewidth=2)
                if len(self._plot_val_total_losses) > 0:
                    plt.plot(range(1, len(self._plot_val_total_losses) + 1), self._plot_val_total_losses, 'r-', label='Valid Total', linewidth=2)
                plt.title('SSL v4: Total Loss (MSE + CE)')
                plt.xlabel('Epoch'); plt.ylabel('Total Loss')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join('logs', 'loss_curves_v4_total.png'))
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
