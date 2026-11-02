import copy
import functools
import math
import os
import lightning as pl
import torch
import torch._inductor.config
import stable_pretraining as spt
from einops import rearrange

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from stable_pretraining.backbone.utils import TeacherStudentWrapper

import hydra
from omegaconf import OmegaConf

from data import (Episodes, WindowedLanceDataset, VariableLengthBatchSampler, ViewBuilder, identity_collate, holdout_episodes,
                  holdout_batches, CachedBatches)
from model import Encoder, Projector
from loss import SIGReg


# Muon only updates the 2-D weight matrices inside the transformer blocks; everything else (patch embedding, readout
# token, norms, projector) stays on AdamW. (student\.)?: with an EMA the encoder is wrapped, and its teacher gets no
# gradient
MUON_MODULES = r"encoder\.(student\.)?blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate|up|down))$"

def WSD(optimizer, total_steps, peak_step, start_factor=0.01, decay_steps=0, end_factor=0.0):
    """Warmup-stable-decay: linear warmup from start_factor * lr to lr over `peak_step` steps, flat, then the 1-sqrt
    cooldown lr * (1 - sqrt(progress)) (Hägele et al. 2024) down to end_factor * lr over the last `decay_steps` of
    `total_steps`. decay_steps=0 is the stable phase alone. To branch a longer decay off a stable-phase checkpoint,
    resume it with max_epochs set to its epoch + decay_epochs: the restored scheduler keeps its step count but takes the
    factor from the new config."""
    decay_start = total_steps - decay_steps

    def factor(step):
        if step < peak_step:
            return start_factor + (1 - start_factor) * step / peak_step
        if step < decay_start or decay_steps == 0:
            return 1.0
        progress = min(1.0, (step - decay_start) / decay_steps)
        return end_factor + (1 - end_factor) * (1 - math.sqrt(progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

def keep_tokens(fraction, size, patch_size):
    """Patch tokens kept per frame of a `size`px view: `fraction` of them, rounded up to a whole token."""
    assert 0 < fraction <= 1, f"masking.keep_* is the fraction of patch tokens kept per frame, got {fraction}"
    return math.ceil(fraction * (size // patch_size) ** 2 - 1e-9)   # the epsilon keeps exact products from rounding up

def build_scheduler(cfg, total_steps):
    params = OmegaConf.to_container(cfg.scheduler)
    if params["type"] != "WSD":
        return {**params, "total_steps": total_steps}
    params.pop("type")
    decay_steps = round(params.pop("decay_epochs", 0) * cfg.data.batches_per_epoch)
    return functools.partial(WSD, total_steps=total_steps, decay_steps=decay_steps, **params)

def build_optim(cfg, total_steps):
    adamw = {"optimizer": OmegaConf.to_container(cfg.optimizer),
             "scheduler": build_scheduler(cfg, total_steps),
             "interval": cfg.interval}
    if cfg.get("muon") is None:
        return adamw
    # first match wins: block matrices go to Muon, all remaining parameters to AdamW
    muon = {**adamw, "optimizer": {"type": "Muon", **OmegaConf.to_container(cfg.muon)}}
    return {"muon": {"modules": MUON_MODULES, **muon}, "adamw": {"modules": ".*", **adamw}}

def forward(self, batch, stage):
    if stage == "fit":
        return loss_step(self, batch, "train")
    # holdout: the same clips every time, and per batch the same crops / patch masks / SIGReg directions, so the curve
    # only moves with the weights; the forked RNG leaves the training stream untouched
    with torch.random.fork_rng(devices=[batch["clip"].device]):
        torch.manual_seed(batch["seed"])
        return loss_step(self, batch, "val")

def loss_step(self, batch, split):
    globals, locals = self.view_builder(batch["clip"])
    V = locals.shape[0]
    encoder = getattr(self.encoder, "student", self.encoder)  # the EMA teacher only serves evaluation

    g_read, _, _ = encoder(globals, keep_per_frame=self.keep_global, iid=self.iid)

    locals = rearrange(locals, "v b t c h w -> (v b) t c h w")
    l_read, _, _ = encoder(locals, keep_per_frame=self.keep_local, iid=self.iid)
    l_read = rearrange(l_read, "(v b) d -> v b d", v=V)

    proj = self.projector(torch.cat([g_read[None], l_read]))
    g_proj, l_proj = proj[0], proj[1:]

    # every local view predicts the global view's embedding; SIGReg keeps all views' embeddings isotropic Gaussian
    invariance = (g_proj - l_proj).pow(2).mean()
    sig = self.sigreg(proj)

    batch["loss"] = invariance + self.lam * sig
    if split == "train":
        self.log_dict({"train/inv": invariance, "train/sig": sig, "train/T": float(globals.shape[1])}, on_step=True, sync_dist=False)
    else:  # mean over all holdout batches of all ranks, logged once per holdout pass
        self.log_dict({"val/inv": invariance, "val/sig": sig, "val/loss": batch["loss"]}, on_step=False, on_epoch=True, sync_dist=True)
    return batch

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    pl.seed_everything(cfg.seed + rank)

    # a mixture lists its stores under dataset.sources; a single store is a mixture of one
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    sources = dataset_cfg.pop("sources") if "sources" in dataset_cfg else [dataset_cfg]
    episodes = Episodes(sources)
    dataset = WindowedLanceDataset([s["path"] for s in episodes.sources], episodes.offsets,
                                   height=cfg.data.height, width=cfg.data.width,
                                   column_name=cfg.dataset.get("column", "frame"))

    # holdout: `videos` training videos never trained on; their loss is logged as val/* every `every_n_steps` steps
    holdout = cfg.get("holdout")
    if holdout is not None:
        holdout_length = max(int(k) for k in cfg.data.probs)
        holdout_eps = holdout_episodes(episodes, holdout.videos, holdout_length, seed=cfg.seed)

    sampler = VariableLengthBatchSampler(episodes, batch_size=cfg.data.batch_size,
                                         probs={int(k): v for k, v in cfg.data.probs.items()},
                                         batches_per_epoch=cfg.data.batches_per_epoch,
                                         seed=cfg.seed, rank=rank, world_size=world_size,
                                         exclude=holdout_eps if holdout is not None else None)

    # forkserver: the holdout decodes lance in this process before the train workers start, and a forked worker would
    # inherit its threads broken
    train_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=identity_collate,
                                               num_workers=cfg.data.num_workers, pin_memory=cfg.data.num_workers > 0, persistent_workers=cfg.data.persistent_workers,
                                               multiprocessing_context="forkserver" if cfg.data.num_workers > 0 else None)

    if holdout is None:
        data = spt.data.DataModule(train=train_loader)
    else:
        # a few fixed batches per rank, decoded once and then served from RAM by the training process; a copy, so the
        # lance handles it opens here are never the ones the train dataset is pickled to its workers with
        val_loader = torch.utils.data.DataLoader(
            CachedBatches(copy.copy(dataset)), collate_fn=identity_collate, num_workers=0,
            batch_sampler=holdout_batches(episodes, holdout_eps, holdout_length, cfg.data.batch_size, rank, world_size))
        data = spt.data.DataModule(train=train_loader, val=val_loader)

    encoder = Encoder(**cfg.model)
    if cfg.get("ema") is not None:
        # an EMA copy of the encoder (encoder.teacher.* in the checkpoint, encoder.student.* is the trained one),
        # updated after every optimizer step by the TeacherStudentCallback spt adds on its own; equal base and final
        # coefficients turn off spt's cosine ramp towards a frozen teacher
        encoder = TeacherStudentWrapper(encoder, base_ema_coefficient=cfg.ema.decay, final_ema_coefficient=cfg.ema.decay)
    if cfg.get("compile", True):
        # draw the patch-masking torch.rand with the eager RNG, so compiled runs keep exactly the same patches
        torch._inductor.config.fallback_random = True
        # in place, so state_dict keys stay unprefixed. Static shapes: the global and the local views each get their
        # own graph instead of a dynamic-shape recompile on the second shape. After wrapping: the teacher is a
        # deepcopy, and only the student runs in training
        getattr(encoder, "student", encoder).compile(dynamic=False)

    patch_size = cfg.model.get("patch_size", 16)
    keep_global = keep_tokens(cfg.masking.keep_global, cfg.views.global_size, patch_size)
    keep_local = keep_tokens(cfg.masking.keep_local, cfg.views.local_size, patch_size)
    print(f"keeping {keep_global} of {(cfg.views.global_size // patch_size) ** 2} global and {keep_local} of "
          f"{(cfg.views.local_size // patch_size) ** 2} local patch tokens per frame", flush=True)

    module = spt.Module(
        forward=forward,
        view_builder=ViewBuilder(**cfg.views),
        encoder=encoder,
        projector=Projector(cfg.model.residual_dim, cfg.projector.hidden, cfg.projector.out),
        sigreg=SIGReg(),
        keep_global=keep_global, keep_local=keep_local,
        iid=cfg.masking.get("iid", False), lam=cfg.loss.lam,
        optim=build_optim(cfg, total_steps=cfg.trainer.max_epochs * cfg.data.batches_per_epoch),
    )

    logger = WandbLogger(**cfg.wandb, save_dir="wandb", log_model=False)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [ModelCheckpoint(**OmegaConf.to_container(cfg.checkpoint, resolve=True))] if "checkpoint" in cfg else []
    val_kwargs = {"val_check_interval": holdout.every_n_steps} if holdout is not None else {}
    trainer = pl.Trainer(**cfg.trainer, **val_kwargs, logger=logger, callbacks=callbacks)
    if cfg.get("gradient_clip_val") is not None:
        # spt.Module uses manual optimization, so Lightning rejects Trainer(gradient_clip_val=...); spt instead reads
        # these trailing-underscore attributes off the trainer and clips before every optimizer step
        trainer.gradient_clip_val_ = float(cfg.gradient_clip_val)
        trainer.gradient_clip_algorithm_ = "norm"
    # resume: a checkpoint to continue from with its full state (weights, EMA, optimizer, step count)
    spt.Manager(trainer=trainer, module=module, data=data, seed=cfg.seed + rank,
                ckpt_path=cfg.get("resume"), weights_only=False)()

if __name__ == "__main__":
    main()