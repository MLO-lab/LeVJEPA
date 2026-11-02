import os
import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import stable_pretraining as spt
from einops import rearrange

from lightning.pytorch.loggers import WandbLogger

import hydra
from omegaconf import ListConfig, OmegaConf

from data import identity_collate
from model import Encoder
from legacy.encoder import LegacyEncoder, load_legacy_encoder
from eval.probe import AttentiveClassifier, CosineLRWD, param_groups

def encode(self, clips):
    """Normalised clips (B, T, C, H, W) -> the probe's tokens: every frame's patches (B, T*N, D) or the last frame's."""
    with torch.no_grad():
        _, patches, _ = self.encoder(clips)
    return patches[:, -1] if self.last_frame else rearrange(patches, "b t n d -> b (t n) d")

def forward(self, batch, stage):
    label = batch["label"]
    if "image" in batch:
        # V-JEPA image protocol: the dataset augmented and normalised the image, which is repeated over the frames
        tokens = encode(self, batch["image"][:, None].expand(-1, self.num_frames, -1, -1, -1))
        views = 1
    else:
        # V-JEPA video protocol: every segment of every spatial view is encoded on its own, and the probe attends across
        # the segments of a view (attend_across_segments), one prediction per view
        views, S = batch["segments"].shape[1:3]
        tokens = encode(self, rearrange(batch["segments"], "b v s c t h w -> (b v s) t c h w"))
        tokens = rearrange(tokens, "(bv s) n d -> bv (s n) d", s=S)

    # one probe per learning rate on the same frozen tokens; each is trained by its own loss only
    losses = []
    for tag, probe, top1 in zip(self.tags, self.probes, self.top1):
        # (B, views, classes): the loss averages over views, the prediction is the mean softmax over views (V-JEPA)
        logits = rearrange(probe(tokens), "(b v) k -> b v k", v=views)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), label.repeat_interleave(views))
        probs = logits.float().softmax(-1).mean(1)
        if stage == "fit":
            acc = (probs.argmax(1) == label).float().mean()
            self.log_dict({f"train/loss@{tag}": loss, f"train/top1@{tag}": acc}, on_step=True, on_epoch=False)
        else:
            top1(probs, label)
            self.log_dict({f"val/loss@{tag}": loss, f"val/top1@{tag}": top1}, on_step=False, on_epoch=True, sync_dist=True)
        losses.append(loss)
    batch["loss"] = torch.stack(losses).sum()
    return batch

class BestProbe(pl.Callback):
    """Tracks the best val/top1 over epochs and learning rates, logs it as val/best_top1, and stops training once no
    probe has improved on it for `patience` validation epochs (None: never stop early)."""

    def __init__(self, tags, patience=None):
        self.tags, self.patience = tags, patience
        self.best, self.best_tag, self.best_epoch, self.wait = -1.0, None, -1, 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        scores = {tag: float(trainer.callback_metrics[f"val/top1@{tag}"]) for tag in self.tags}
        tag = max(scores, key=scores.get)
        if scores[tag] > self.best:
            self.best, self.best_tag, self.best_epoch, self.wait = scores[tag], tag, trainer.current_epoch, 0
        else:
            self.wait += 1
        if trainer.logger is not None:
            trainer.logger.log_metrics({"val/top1": scores[tag], "val/best_top1": self.best}, step=trainer.global_step)
        if self.patience is not None and self.wait >= self.patience:
            trainer.should_stop = True   # every rank sees the same synced metrics, so they stop together

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero and self.best_tag is not None:
            print(f"best val/top1 {self.best:.4f} at lr {self.best_tag}, epoch {self.best_epoch}", flush=True)

def load_encoder(cfg):
    if cfg.model.get("legacy"):  # a LeVJEPA v1 checkpoint, see legacy/encoder.py
        if cfg.checkpoint is None:
            encoder = LegacyEncoder(**cfg.model)
        else:
            print(f"loading v1 encoder from {cfg.checkpoint}", flush=True)
            encoder = load_legacy_encoder(cfg.checkpoint, ema=cfg.get("ema", True), **cfg.model)
    else:
        encoder = load_v2_encoder(cfg)
    encoder.requires_grad_(False)
    if cfg.get("compile", True):
        # in place (state_dict keys unchanged), one static graph per clip shape
        encoder.compile(dynamic=False)
    return encoder

def load_v2_encoder(cfg):
    encoder = Encoder(**cfg.model)
    if cfg.checkpoint is not None:
        print(f"loading encoder from {cfg.checkpoint}", flush=True)
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        prefix = "encoder."
        if any(k.startswith("encoder.teacher.") for k in state):  # trained with an EMA: teacher by default
            prefix = "encoder.teacher." if cfg.get("ema", True) else "encoder.student."
            print(f"loading the {'EMA teacher' if cfg.get('ema', True) else 'student'} encoder", flush=True)
        encoder.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)})
    return encoder

def loader(dataset, cfg, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=shuffle, drop_last=shuffle, collate_fn=identity_collate,
                                       num_workers=cfg.data.num_workers, pin_memory=True, persistent_workers=cfg.data.num_workers > 0,
                                       multiprocessing_context="spawn" if cfg.data.num_workers > 0 else None)

@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg):
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    pl.seed_everything(cfg.seed + rank)

    train_set = hydra.utils.instantiate(cfg.benchmark.train, num_frames=cfg.benchmark.num_frames, size=cfg.data.size)
    val_set = hydra.utils.instantiate(cfg.benchmark.val, num_frames=cfg.benchmark.num_frames, size=cfg.data.size)
    data = spt.data.DataModule(train=loader(train_set, cfg, shuffle=True), val=loader(val_set, cfg, shuffle=False))
    total_steps = cfg.trainer.max_epochs * (len(train_set) // (cfg.data.batch_size * world_size))

    # optimizer.lr: one learning rate, or a list to search, each trained as its own probe in the same run
    opt = cfg.optimizer
    lrs = list(opt.lr) if isinstance(opt.lr, ListConfig) else [opt.lr]
    probes = torch.nn.ModuleList(AttentiveClassifier(cfg.model.residual_dim, cfg.probe.num_heads, cfg.benchmark.num_classes)
                                 for _ in lrs)
    tags = [f"{lr:g}" for lr in lrs]
    module = spt.Module(
        forward=forward,
        encoder=load_encoder(cfg),
        probes=probes,
        top1=torch.nn.ModuleList(torchmetrics.classification.MulticlassAccuracy(cfg.benchmark.num_classes, average="micro")
                                 for _ in lrs),
        tags=tags,
        last_frame=cfg.probe.frames == "last",
        num_frames=cfg.benchmark.num_frames,
        # V-JEPA 2's init_opt: AdamW (default betas) without weight decay on biases and 1-D params, per-step cosine lr
        # and wd; one optimizer per probe
        optim={f"probe{i}": {"modules": rf"probes\.{i}$",
                             "optimizer": lambda _, p=p, lr=lr: torch.optim.AdamW(param_groups(p), lr=lr, weight_decay=opt.weight_decay),
                             "scheduler": lambda optimizer: CosineLRWD(optimizer, total_steps, opt.final_weight_decay),
                             "interval": "step"}
               for i, (p, lr) in enumerate(zip(probes, lrs))},
    )

    # spt resumes the wandb run named in ./wandb_resume.json, which every run from this checkout rewrites; evals never
    # resume, so drop it or this run would log into another run
    try:
        os.remove("wandb_resume.json")
    except FileNotFoundError:   # absent, or another rank removed it first
        pass
    logger = WandbLogger(**cfg.wandb, save_dir="wandb", log_model=False)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [BestProbe(tags, cfg.early_stopping.patience if cfg.get("early_stopping") else None)]
    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    if cfg.get("gradient_clip_val") is not None:
        trainer.gradient_clip_val_ = float(cfg.gradient_clip_val)
        trainer.gradient_clip_algorithm_ = "norm"
    spt.Manager(trainer=trainer, module=module, data=data, seed=cfg.seed + rank)()

if __name__ == "__main__":
    main()
