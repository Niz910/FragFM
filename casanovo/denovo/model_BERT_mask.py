"""A mass spectrometry data encoding model with BERT-style m/z prediction and classification."""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from casanovo import config
from casanovo.denovo.transformers import SpectrumEncoder


logger = logging.getLogger("casanovo")


class MSEncoder(pl.LightningModule):
    """
    A Transformer model for encoding mass spectrometry data and predicting m/z values.
    
    This model uses BERT-style masking to predict masked m/z values in mass spectra.
    It supports both regression (continuous m/z prediction) and classification 
    (discretized m/z bin prediction) tasks.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must
        be divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the
        transformer model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The
        remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value. If ``None``, the intensity will be
        projected up to ``dim_model`` using a linear layer, then summed
        with the m/z encoding for each peak.
    max_peaks : int
        The maximum number of peaks in a spectrum.
    mask_prob : float
        The probability of masking m/z values during training.
    bin_size : float
        The size of each m/z bin for classification (in Da).
    max_mz : float
        The maximum m/z value to consider for binning.
    classification_weight : float
        The weight for classification loss in the combined loss function.
    regression_weight : float
        The weight for regression loss in the combined loss function.
    n_log : int
        The number of epochs to wait between logging messages.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_peaks: int = 1000,
        mask_prob: float = 0.15,
        bin_size: float = 1.0,
        max_mz: float = 2000.0,
        classification_weight: float = 0.5,
        regression_weight: float = 0.5,
        n_log: int = 10,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build the model.
        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        
        # M/Z regression prediction head
        self.mz_predictor = nn.Linear(dim_model, 1)
        
        # M/Z classification prediction head
        self.n_bins = int(max_mz / bin_size) + 1  # +1 for the last bin
        self.mz_classifier = nn.Linear(dim_model, self.n_bins)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)  # -1 for invalid bins
        
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        
        # `kwargs` will contain additional arguments as well as
        # unrecognized arguments, including deprecated ones. Remove the
        # deprecated ones.
        for k in config._config_deprecated:
            kwargs.pop(k, None)
            warnings.warn(
                f"Deprecated hyperparameter '{k}' removed from the model.",
                DeprecationWarning,
            )
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_peaks = max_peaks
        self.mask_prob = mask_prob
        self.bin_size = bin_size
        self.max_mz = max_mz
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight

        # Logging.
        self.n_log = n_log
        self._history = []
        # Lists for plotting loss curves
        self._plot_train_losses = []
        self._plot_val_losses = []

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is currently running.

        Returns
        -------
        torch.device
            The device on which the model is currently running.
        """
        return next(self.parameters()).device

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode mass spectrometry data and predict m/z values.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.

        Returns
        -------
        encoded_features : torch.Tensor of shape (batch_size, max_peaks, dim_model)
            Encoded features for each peak in the spectrum.
        predicted_mz : torch.Tensor of shape (batch_size, max_peaks)
            Predicted m/z values for each peak (regression).
        predicted_bins : torch.Tensor of shape (batch_size, max_peaks, n_bins)
            Predicted m/z bin probabilities for each peak (classification).
        mask_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks were masked (replaced with 0).
        random_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks were replaced with random values.
        original_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks were kept original.
        """
        mzs, intensities, precursors = self._process_batch(batch)
        
        # Apply BERT-style masking
        masked_mzs, mask_tokens, random_tokens, original_tokens = self._apply_bert_masking(mzs)
        
        # Encode the spectrum using masked m/z values
        memories, mem_masks = self.encoder(masked_mzs, intensities)
        
        # Predict m/z values (regression)
        predicted_mz = self.mz_predictor(memories).squeeze(-1)
        
        # Predict m/z bins (classification)
        predicted_bins = self.mz_classifier(memories)
        
        return memories, predicted_mz, predicted_bins, mask_tokens, random_tokens, original_tokens

    def _create_mask(self, mzs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create BERT-style mask for training: 80% mask, 10% random, 10% original.
        
        Parameters
        ----------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            The m/z values of spectra.
            
        Returns
        -------
        mask_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks to mask (replace with 0).
        random_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks to replace with random values.
        original_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks to keep original.
        """
        batch_size, max_peaks = mzs.shape
        device = mzs.device
        
        # Only consider valid peaks (non-zero m/z values)
        valid_peaks = mzs > 0
        
        # Create random probabilities for each position
        rand_probs = torch.rand(batch_size, max_peaks, device=device)
        
        # Initialize masks
        mask_tokens = torch.zeros_like(valid_peaks, dtype=torch.bool, device=device)
        random_tokens = torch.zeros_like(valid_peaks, dtype=torch.bool, device=device)
        original_tokens = torch.zeros_like(valid_peaks, dtype=torch.bool, device=device)
        
        # Apply BERT-style masking only to valid peaks
        valid_indices = torch.where(valid_peaks)
        
        if len(valid_indices[0]) > 0:
            # Get random probabilities for valid positions
            valid_rand_probs = rand_probs[valid_indices]
            
            # 80% mask (replace with 0)
            mask_80 = valid_rand_probs < 0.8
            mask_tokens[valid_indices] = mask_80
            
            # 10% random (replace with random m/z value)
            random_10 = (valid_rand_probs >= 0.8) & (valid_rand_probs < 0.9)
            random_tokens[valid_indices] = random_10
            
            # 10% original (keep original value)
            original_10 = valid_rand_probs >= 0.9
            original_tokens[valid_indices] = original_10
        
        return mask_tokens, random_tokens, original_tokens

    def _generate_random_mz(self, mzs: torch.Tensor, random_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate random m/z values for the random replacement strategy.
        
        Parameters
        ----------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            The original m/z values of spectra.
        random_mask : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks to replace with random values.
            
        Returns
        -------
        random_mzs : torch.Tensor of shape (batch_size, max_peaks)
            Random m/z values for the masked positions.
        """
        batch_size, max_peaks = mzs.shape
        device = mzs.device
        
        # Generate random m/z values in the range [50, max_mz]
        min_mz = 50.0  # Minimum reasonable m/z value
        random_mzs = torch.rand(batch_size, max_peaks, device=device) * (self.max_mz - min_mz) + min_mz
        
        # Only apply random values where the mask is True
        random_mzs = random_mzs * random_mask.float()
        
        return random_mzs

    def _apply_bert_masking(self, mzs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply BERT-style masking to m/z values.
        
        Parameters
        ----------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            The original m/z values of spectra.
            
        Returns
        -------
        masked_mzs : torch.Tensor of shape (batch_size, max_peaks)
            M/z values after applying BERT-style masking.
        mask_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks were masked.
        random_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks were replaced with random values.
        original_tokens : torch.Tensor of shape (batch_size, max_peaks)
            Boolean mask indicating which peaks were kept original.
        """
        # Create BERT-style masks
        mask_tokens, random_tokens, original_tokens = self._create_mask(mzs)
        
        # Start with original m/z values
        masked_mzs = mzs.clone()
        
        # Apply masking: set masked positions to 0
        masked_mzs[mask_tokens] = 0
        
        # Apply random replacement
        if random_tokens.any():
            random_mzs = self._generate_random_mz(mzs, random_tokens)
            masked_mzs[random_tokens] = random_mzs[random_tokens]
        
        # Original tokens are already correct (no change needed)
        
        return masked_mzs, mask_tokens, random_tokens, original_tokens

    def _mz_to_bins(self, mzs: torch.Tensor) -> torch.Tensor:
        """
        Convert m/z values to bin indices using fixed bin width.
        
        Parameters
        ----------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            The m/z values of spectra.
            
        Returns
        -------
        bins : torch.Tensor of shape (batch_size, max_peaks)
            Bin indices for each m/z value. -1 for invalid/out-of-range values.
        """
        # Use fixed bin width
        bin_width = 0.1
        
        # Convert to bin indices using fixed width
        bins = torch.floor(mzs / bin_width).long()
        
        # Handle invalid values
        invalid_mask = (mzs <= 0) | (mzs > self.max_mz)
        bins[invalid_mask] = -1
        
        # Ensure we don't exceed n_bins
        bins = torch.clamp(bins, min=-1, max=self.n_bins - 1)
        
        return bins

    def _bins_to_mz(self, bins: torch.Tensor) -> torch.Tensor:
        """
        Convert bin indices back to m/z values.
        
        Parameters
        ----------
        bins : torch.Tensor of shape (batch_size, max_peaks)
            Bin indices.
            
        Returns
        -------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            Corresponding m/z values.
        """
        # Convert bins to m/z values (center of bin)
        mzs = (bins.float() + 0.5) * self.bin_size
        
        # Set invalid bins to 0
        invalid_mask = bins == -1
        mzs[invalid_mask] = 0
        
        return mzs

    def _process_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a SpectrumDataset batch to tensors.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.

        Returns
        -------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            The m/z values for each spectrum.
        intensities : torch.Tensor of shape (batch_size, max_peaks)
            The intensity values for each spectrum.
        precursors : torch.Tensor of shape (batch_size, 3)
            A tensor with the precursor neutral mass, precursor charge,
            and precursor m/z.
        """
        precursor_mzs = batch["precursor_mz"].squeeze(0)
        precursor_charges = batch["precursor_charge"].squeeze(0)
        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
        precursors = torch.vstack(
            [precursor_masses, precursor_charges, precursor_mzs]
        ).T

        mzs = batch["mz_array"]
        intensities = batch["intensity_array"]

        return mzs, intensities, precursors

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        mzs, intensities, precursors = self._process_batch(batch)
        
        # Apply BERT-style masking
        masked_mzs, mask_tokens, random_tokens, original_tokens = self._apply_bert_masking(mzs)
        
        # Encode the spectrum using masked m/z values
        memories, mem_masks = self.encoder(masked_mzs, intensities)
        
        # Predict m/z values (regression)
        predicted_mz = self.mz_predictor(memories).squeeze(-1)
        
        # Handle dimension mismatch if it occurs
        min_len = min(predicted_mz.size(1), mzs.size(1))
        if predicted_mz.size(1) != mzs.size(1):
            predicted_mz = predicted_mz[:, :min_len]
            mzs = mzs[:, :min_len]
            intensities = intensities[:, :min_len]
            mask_tokens = mask_tokens[:, :min_len]
            random_tokens = random_tokens[:, :min_len]
            original_tokens = original_tokens[:, :min_len]
            memories = memories[:, :min_len]
            
        # Predict m/z bins (classification)
        predicted_bins = self.mz_classifier(memories)
        
        # Convert true m/z values to bins
        true_bins = self._mz_to_bins(mzs)
        
        # Create combined mask for loss calculation (mask + random + original)
        # We train on all three types: masked, random, and original
        train_mask = mask_tokens | random_tokens | original_tokens
        
        # Calculate regression loss on all training positions
        regression_loss = self.mse_loss(predicted_mz[train_mask], mzs[train_mask])
        
        # Calculate classification loss on all training positions
        classification_loss = self.ce_loss(
            predicted_bins[train_mask].view(-1, self.n_bins), 
            true_bins[train_mask].view(-1)
        )
        
        # Combined loss
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss)
        
        # Log individual losses and masking statistics
        self.log(
            f"{mode}_MSE_Loss",
            regression_loss.detach(),
            on_step=True,  # 改为每步都记录
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
            prog_bar=True,  # 在进度条显示
        )
        self.log(
            f"{mode}_CE_Loss",
            classification_loss.detach(),
            on_step=True,  # 改为每步都记录
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
            prog_bar=True,  # 在进度条显示
        )
        self.log(
            f"{mode}_Total_Loss",
            total_loss.detach(),
            on_step=True,  # 改为每步都记录
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
            prog_bar=True,  # 在进度条显示
        )

        # Also log a generic train_loss key so external training scripts
        # that monitor "train_loss" (like train_ssl_mgf.py) will find it.
        if mode == "train":
            self.log(
                "train_loss",
                total_loss.detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=mzs.shape[0],
            )
        
        # Log masking statistics
        mask_ratio = mask_tokens.float().mean()
        random_ratio = random_tokens.float().mean()
        original_ratio = original_tokens.float().mean()
        
        self.log(
            f"{mode}_Mask_Ratio",
            mask_ratio.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        self.log(
            f"{mode}_Random_Ratio",
            random_ratio.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        self.log(
            f"{mode}_Original_Ratio",
            original_ratio.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        
        return total_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        mzs, intensities, precursors = self._process_batch(batch)
        
        # Apply BERT-style masking
        masked_mzs, mask_tokens, random_tokens, original_tokens = self._apply_bert_masking(mzs)
        
        # Encode the spectrum using masked m/z values
        memories, mem_masks = self.encoder(masked_mzs, intensities)
        
        # Predict m/z values (regression)
        predicted_mz = self.mz_predictor(memories).squeeze(-1)
        
        # Handle dimension mismatch if it occurs
        min_len = min(predicted_mz.size(1), mzs.size(1))
        if predicted_mz.size(1) != mzs.size(1):
            predicted_mz = predicted_mz[:, :min_len]
            mzs = mzs[:, :min_len]
            intensities = intensities[:, :min_len]
            mask_tokens = mask_tokens[:, :min_len]
            random_tokens = random_tokens[:, :min_len]
            original_tokens = original_tokens[:, :min_len]
            memories = memories[:, :min_len]
            
        # Predict m/z bins (classification)
        predicted_bins = self.mz_classifier(memories)
        
        # Convert true m/z values to bins
        true_bins = self._mz_to_bins(mzs)
        
        # Create combined mask for loss calculation (mask + random + original)
        train_mask = mask_tokens | random_tokens | original_tokens
        
        # Calculate regression loss on all training positions
        regression_loss = self.mse_loss(predicted_mz[train_mask], mzs[train_mask])
        
        # Calculate classification loss on all training positions
        classification_loss = self.ce_loss(
            predicted_bins[train_mask].view(-1, self.n_bins), 
            true_bins[train_mask].view(-1)
        )
        
        # Combined loss
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss)
        
        # Calculate classification accuracy
        predicted_bin_indices = torch.argmax(predicted_bins[train_mask], dim=-1)
        true_bin_indices = true_bins[train_mask]
        valid_mask = true_bin_indices != -1
        if valid_mask.sum() > 0:
            accuracy = (predicted_bin_indices[valid_mask] == true_bin_indices[valid_mask]).float().mean()
        else:
            accuracy = torch.tensor(0.0, device=self.device)
        
        # Log individual losses and accuracy
        self.log(
            "valid_MSE_Loss",
            regression_loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        self.log(
            "valid_CE_Loss",
            classification_loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        self.log(
            "valid_Total_Loss",
            total_loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        # Also log a generic val_loss key to match common training scripts
        self.log(
            "val_loss",
            total_loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        self.log(
            "valid_Accuracy",
            accuracy.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        
        # Log masking statistics for validation
        mask_ratio = mask_tokens.float().mean()
        random_ratio = random_tokens.float().mean()
        original_ratio = original_tokens.float().mean()
        
        self.log(
            "valid_Mask_Ratio",
            mask_ratio.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        self.log(
            "valid_Random_Ratio",
            random_ratio.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        self.log(
            "valid_Original_Ratio",
            original_ratio.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mzs.shape[0],
        )
        
        return total_loss

    def on_train_epoch_end(self) -> None:
        """
        Log the training metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        train_loss = callback_metrics.get("train_Total_Loss_epoch", torch.tensor(np.nan)).detach().item()
        train_mse = callback_metrics.get("train_MSE_Loss_epoch", torch.tensor(np.nan)).detach().item()
        train_ce = callback_metrics.get("train_CE_Loss_epoch", torch.tensor(np.nan)).detach().item()
        mask_ratio = callback_metrics.get("train_Mask_Ratio", torch.tensor(np.nan)).detach().item()
        random_ratio = callback_metrics.get("train_Random_Ratio", torch.tensor(np.nan)).detach().item()
        original_ratio = callback_metrics.get("train_Original_Ratio", torch.tensor(np.nan)).detach().item()
        
        # 每个epoch结束时打印详细信息
        print(f"\nEpoch {self.current_epoch} Summary:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  MSE Loss: {train_mse:.4f}")
        print(f"  CE Loss: {train_ce:.4f}")
        print(f"  Mask/Random/Original Ratios: {mask_ratio:.2%}/{random_ratio:.2%}/{original_ratio:.2%}")
        
        metrics = {
            "step": self.trainer.global_step, 
            "train_total": train_loss,
            "train_mse": train_mse,
            "train_ce": train_ce,
            "mask_ratio": mask_ratio,
            "random_ratio": random_ratio,
            "original_ratio": original_ratio
        }
        self._history.append(metrics)
        # Record train loss for plotting (may be nan)
        try:
            self._plot_train_losses.append(metrics.get("train_total", float('nan')))
        except Exception:
            self._plot_train_losses.append(float('nan'))
        self._log_history()
        # Update plots
        try:
            self._plot_loss_curves()
        except Exception as e:
            logger.warning(f"Failed to plot train loss curves: {e}")

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        valid_loss = callback_metrics.get("valid_Total_Loss", torch.tensor(np.nan)).detach().item()
        valid_mse = callback_metrics.get("valid_MSE_Loss", torch.tensor(np.nan)).detach().item()
        valid_ce = callback_metrics.get("valid_CE_Loss", torch.tensor(np.nan)).detach().item()
        valid_acc = callback_metrics.get("valid_Accuracy", torch.tensor(np.nan)).detach().item()
        
        metrics = {
            "step": self.trainer.global_step,
            "valid_total": valid_loss,
            "valid_mse": valid_mse,
            "valid_ce": valid_ce,
            "valid_acc": valid_acc
        }
        self._history.append(metrics)
        # Record val loss for plotting
        try:
            self._plot_val_losses.append(metrics.get("valid_total", float('nan')))
        except Exception:
            self._plot_val_losses.append(float('nan'))
        self._log_history()
        # Update plots
        try:
            self._plot_loss_curves()
        except Exception as e:
            logger.warning(f"Failed to plot val loss curves: {e}")

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain Total\tTrain MSE\tTrain CE\tValid Total\tValid MSE\tValid CE\tValid Acc\tMask%\tRandom%\tOrig%"
            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.2f\t%.2f\t%.2f"
            vals = [
                metrics["step"],
                metrics.get("train_total", np.nan),
                metrics.get("train_mse", np.nan),
                metrics.get("train_ce", np.nan),
                metrics.get("valid_total", np.nan),
                metrics.get("valid_mse", np.nan),
                metrics.get("valid_ce", np.nan),
                metrics.get("valid_acc", np.nan),
                metrics.get("mask_ratio", np.nan) * 100,
                metrics.get("random_ratio", np.nan) * 100,
                metrics.get("original_ratio", np.nan) * 100,
            ]
            logger.info(msg, *vals)

    def _plot_loss_curves(self) -> None:
        """Plot and save training and validation loss curves to logs/loss_curves_mse.png."""
        try:
            # If no data yet, skip
            if len(self._plot_train_losses) == 0 and len(self._plot_val_losses) == 0:
                return

            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self._plot_train_losses) + 1)
            if len(self._plot_train_losses) > 0:
                plt.plot(epochs, self._plot_train_losses, 'b-', label='Train Total Loss')

            if len(self._plot_val_losses) > 0:
                # Align val plot to its own length in case val and train counts differ
                v_epochs = range(1, len(self._plot_val_losses) + 1)
                plt.plot(v_epochs, self._plot_val_losses, 'r-', label='Valid Total Loss')

            plt.title('Loss Curves (MSE Encoder)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            os.makedirs('logs', exist_ok=True)
            plt.savefig(os.path.join('logs', 'loss_curves_mse.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to create loss curve plot: {e}")

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], Dict[str, Any]]:
        """
        Initialize the optimizer.

        We use the Adam optimizer with a cosine learning rate scheduler.

        Returns
        -------
        Tuple[List[torch.optim.Optimizer], Dict[str, Any]]
            The initialized Adam optimizer and its learning rate
            scheduler.
        """
        # Filter out non-optimizer kwargs to only pass valid Adam parameters
        valid_opt_kwargs = {
            k: v for k, v in self.opt_kwargs.items() 
            if k in ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
        }
        optimizer = torch.optim.Adam(self.parameters(), **valid_opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_iters, self.cosine_schedule_period_iters
        )
        # Configure scheduler to be called after optimizer step
        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "train_loss",
            "strict": False,  # don't fail if metric not found
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine
    shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor


# Backwards-compatible alias: allow importing SpectrumSSL from this module
# if another training script expects that class name.
SpectrumSSL = MSEncoder
