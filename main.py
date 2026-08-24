import hydra
import lightning as pl
import stable_pretraining as spt
import stable_pretraining.optim.utils as spt_optim_utils
import torch
from einops import rearrange
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf

from callbacks import WeightDecayUpdater, WeightEMA
import module as vit_models
from data.loader import build_loader
from module import SIGReg

torch.set_float32_matmul_precision("high")


def _is_bias_or_norm_param(name: str, param: torch.nn.Parameter) -> bool:
    name_lower = name.lower()
    return "bias" in name_lower or "norm" in name_lower or param.ndim == 1


spt_optim_utils.is_bias_or_norm_param = _is_bias_or_norm_param


def build_model(cfg: DictConfig):
    factory = getattr(vit_models, cfg.model.name)
    return factory(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.model.num_frames,
        tubelet_size=cfg.model.tubelet_size,
        use_rope=cfg.model.get("use_rope", True),
        token_drop_rate=cfg.model.get("token_drop_rate", 0.0),
        token_drop_mode=cfg.model.get("token_drop_mode", "random"),
        token_drop_k=cfg.model.get("token_drop_k", 2),
        attn_mode=cfg.model.get("attn_mode", "full"),
    )


def build_projector(cfg: DictConfig, encoder: torch.nn.Module):
    return vit_models.Projector(
        input_dim=encoder.embed_dim,
        hidden_dim=cfg.projector.hidden_dim,
        output_dim=cfg.projector.output_dim,
        norm_layer=torch.nn.BatchNorm1d,
    )


def resolve_data_spec(spec):
    """Resolve `data.<split>` (a path, or a list of source mappings) to absolute paths."""
    if isinstance(spec, (DictConfig, ListConfig)):
        spec = OmegaConf.to_container(spec, resolve=True)
    if isinstance(spec, str):
        return to_absolute_path(spec)
    if isinstance(spec, dict):
        spec = [spec]
    resolved = []
    for entry in spec:
        if isinstance(entry, str):
            entry = {"path": entry}
        else:
            entry = dict(entry)
        entry["path"] = to_absolute_path(entry["path"])
        resolved.append(entry)
    return resolved


def build_video_loader(cfg: DictConfig, split: str, shuffle: bool, hflip: bool):
    aug = cfg.augmentation
    return build_loader(
        resolve_data_spec(cfg.data[split]),
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        num_frames=cfg.model.num_frames,
        frame_stride=cfg.frame_stride,
        crop_size=cfg.model.img_size,
        local_size=aug.local_size,
        local_crops_number=aug.local_crops_number,
        global_crops_scale=aug.global_crops_scale,
        local_crops_scale=aug.local_crops_scale,
        hflip=hflip,
        color_jitter_prob=aug.color_jitter_prob,
        grayscale_prob=aug.grayscale_prob,
        gaussian_blur_prob=aug.gaussian_blur_prob,
        color_jitter_hue=aug.get("color_jitter_hue", 0.1),
        normalize_on_gpu=cfg.loader.get("normalize_on_gpu", False),
        shuffle=shuffle,
        drop_last=split == "train",
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=cfg.loader.persistent_workers,
        prefetch_factor=cfg.loader.prefetch_factor,
        random_crop=split == "train",
        # Small datasets need many clips per video per epoch; val stays at 1.
        clips_per_video=(cfg.loader.get("clips_per_video", 1) if split == "train" else 1),
    )


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def to_float_normalized(x: torch.Tensor) -> torch.Tensor:
    """uint8 -> float32 in [0,1] -> ImageNet-normalized (loader.normalize_on_gpu).

    No-op on float32 input, so runs with normalize_on_gpu=false are unaffected.
    """
    if x.dtype != torch.uint8:
        return x
    x = x.to(torch.float32).div_(255.0)
    mean = torch.as_tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std = torch.as_tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    return x.sub_(mean).div_(std)


def multiview_forward(self, batch, stage):
    global_frame = to_float_normalized(batch["global_frame"])
    local_frames = to_float_normalized(batch["local_frames"])
    batch_size = global_frame.shape[0]

    global_tokens = self.encoder(rearrange(global_frame, "b t c h w -> b c t h w"))
    global_cls = global_tokens[:, 0].unsqueeze(1)

    local_tokens = self.encoder(rearrange(local_frames, "b v t c h w -> (b v) c t h w"))
    local_cls = rearrange(local_tokens[:, 0], "(b v) d -> b v d", b=batch_size)

    embeddings = self.projector(torch.cat([global_cls, local_cls], dim=1))
    global_emb = embeddings[:, :1]

    output = {}
    output["pred_loss"] = (global_emb - embeddings).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(rearrange(embeddings, "b v d -> v b d"))
    output["loss"] = output["pred_loss"] + self.sigreg_weight * output["sigreg_loss"]
    self.log_dict(
        {
            f"{stage}/pred_loss": output["pred_loss"].detach(),
            f"{stage}/sigreg_loss": output["sigreg_loss"].detach(),
            f"{stage}/loss": output["loss"].detach(),
        },
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )

    return output


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    train_loader = build_video_loader(
        cfg,
        split="train",
        shuffle=True,
        hflip=cfg.augmentation.hflip,
    )
    val_loader = build_video_loader(cfg, split="val", shuffle=False, hflip=False)
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    encoder = build_model(cfg)
    projector = build_projector(cfg, encoder)

    world_size = cfg.trainer.devices * cfg.trainer.num_nodes
    # spt.Module does manual optimization, so Lightning rejects
    # Trainer(accumulate_grad_batches>1); spt reads the count from the trainer's
    # `accumulate_grad_batches_` attribute, set after the Trainer is built.
    accum = max(int(cfg.get("accumulate_grad_batches", 1)), 1)
    # scheduler.interval="step" advances per *optimizer* step, so total_steps must
    # be in optimizer-step units (batches / accum).
    batches_per_epoch = len(train_loader.dataset) // world_size // cfg.loader.batch_size
    steps_per_epoch = batches_per_epoch // accum
    total_steps = cfg.trainer.max_epochs * steps_per_epoch

    module = spt.Module(
        encoder=encoder,
        projector=projector,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        sigreg_weight=cfg.loss.sigreg.weight,
        forward=multiview_forward,
        optim={
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": cfg.scheduler.name,
                "total_steps": total_steps,
                "peak_step": cfg.scheduler.peak_step,
                "start_factor": cfg.scheduler.start_lr / cfg.optimizer.lr,
                "end_lr": cfg.scheduler.end_lr,
            },
            "interval": cfg.scheduler.interval,
        },
        hparams=OmegaConf.to_container(cfg, resolve=True),
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = []
    if cfg.weight_decay_scheduler.enabled:
        callbacks.append(
            WeightDecayUpdater(
                schedule_type=cfg.weight_decay_scheduler.schedule_type,
                start_value=cfg.weight_decay_scheduler.start_value,
                end_value=cfg.weight_decay_scheduler.end_value,
                param_group_indices=list(
                    cfg.weight_decay_scheduler.param_group_indices
                ),
            )
        )
    if cfg.ema.enabled:
        ema_cfg = OmegaConf.to_container(cfg.ema, resolve=True)
        ema_cfg.pop("enabled")
        callbacks.append(WeightEMA(**ema_cfg))
    callbacks.append(
        ModelCheckpoint(**OmegaConf.to_container(cfg.checkpoint, resolve=True))
    )
    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    if accum > 1:
        trainer.accumulate_grad_batches_ = accum
        print(f"[grad-accum] spt manual accumulation over {accum} micro-batches "
              f"-> effective batch {cfg.loader.batch_size * world_size * accum}", flush=True)

    # Weights-only continuation
    resume_ckpt_path = cfg.resume.ckpt_path
    if resume_ckpt_path is not None and cfg.resume.weights_only:
        state = torch.load(
            to_absolute_path(resume_ckpt_path), map_location="cpu", weights_only=False
        )
        state_dict = state["state_dict"]
        if getattr(encoder, "pos_embed", None) is None:
            state_dict = dict(state_dict)
            state_dict.pop("encoder.pos_embed", None)
        module.load_state_dict(state_dict, strict=True)
        del state
        print(
            f"[resume] loaded weights from {resume_ckpt_path}; "
            "starting a fresh schedule (ckpt_path=None)",
            flush=True,
        )
        resume_ckpt_path = None

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        ckpt_path=resume_ckpt_path,
        weights_only=cfg.resume.weights_only,
    )
    manager()


if __name__ == "__main__":
    main()
