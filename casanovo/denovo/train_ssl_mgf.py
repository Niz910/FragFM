import os
import types
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from casanovo.denovo.dataloaders import DeNovoDataModule
from casanovo.model_ssl_v1 import SpectrumSSL


def main():
    # =====================================================
    # Configuration
    # =====================================================
    train_mgf = r"C:\Users\Lenovo\Desktop\liucasanovo\sample_data\bacillus.10k.fixed.mgf"
    valid_mgf = None  # optional, 可设为 None

    lance_cache = "lance_cache"
    os.makedirs(lance_cache, exist_ok=True)

    batch_size = 30
    max_epochs = 30
    #n_bins = 64  # 不再使用固定的n_bins，现在在模型中根据mz范围动态计算
    dim_model = 512
    n_head = 8
    n_layers = 6
    lr = 1e-4

    # =====================================================
    # Environment setup
    # =====================================================
    torch.set_float32_matmul_precision("medium")

    # 禁止 Lightning 误判 SLURM 环境（否则会 hang）
    os.environ["PL_DISABLE_SLURM"] = "1"

    # =====================================================
    # Data module
    # =====================================================
    dm = DeNovoDataModule(
    lance_dir=lance_cache,
    train_paths=[train_mgf],
    valid_paths=None,
    train_batch_size=batch_size,
    eval_batch_size=batch_size,
    max_peaks=200,
    min_peaks=20,
    shuffle=True,
    n_workers=8,  # 增加worker数量以提高数据加载性能
)
    #  防报错
    def val_dataloader(self):
        return self.train_dataloader()

    dm.val_dataloader = types.MethodType(val_dataloader, dm)

    # =====================================================
    # Model
    # =====================================================
    model = SpectrumSSL(
        dim_model=dim_model,
        n_head=n_head,
        n_layers=n_layers,
        dim_feedforward=1024,
        dropout=0.1,
        lr=lr,
        #n_bins=n_bins,
        min_mask=0.05,
        max_mask=0.05,
        total_epochs=max_epochs,
        use_cosine_mask_schedule=False,
    )

    # =====================================================
    # Refine stage: freeze encoder after 80% epochs
    # =====================================================
    def on_train_epoch_start(self):
        refine_epoch = int(self.hparams.total_epochs * 0.8)
        if self.current_epoch == refine_epoch:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print(f">>> Entered refine stage (encoder frozen at epoch {refine_epoch})")

    model.on_train_epoch_start = types.MethodType(on_train_epoch_start, model)

    # =====================================================
    # Logging & checkpoint
    # =====================================================
    logger = TensorBoardLogger("logs", name="ssl_mask_binning")

    ckpt_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        filename="epoch{epoch:03d}-train_loss{train_loss:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # =====================================================
    # Trainer
    # =====================================================
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    accelerator = "cuda" if use_gpu else "cpu"
    devices = 1 if use_gpu else None
    precision = "16-mixed" if use_gpu else "32-true"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed",
        strategy="auto",
        logger=logger,
        callbacks=[ckpt_callback, lr_monitor],
        log_every_n_steps=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        limit_val_batches=0,
    )

    # =====================================================
    # Train
    # =====================================================
    print(f"🚀 Starting SSL training on {train_mgf}")
    trainer.fit(model, datamodule=dm)
    print("✅ Training complete.")


if __name__ == "__main__":
    main()