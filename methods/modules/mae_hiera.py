import sys
from typing import Dict, Tuple

import hiera
import hiera.hiera_utils
import torch
import torch.nn.functional as F
from hiera import MaskedAutoencoderHiera, hiera_mae
from hiera.hiera_utils import pretrained_model
from lightly.models import utils
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.dist import print_rank_zero
from lightly.utils.scheduler import CosineWarmupScheduler
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Identity, LayerNorm
from torch.optim import AdamW


@pretrained_model({})
def mae_hiera_base_plus_112(**kwargs):
    return MaskedAutoencoderHiera(
        input_size=(112, 112),
        patch_stride=(2, 2),
        embed_dim=112,
        num_heads=2,
        stages=(2, 3, 16, 3),
        q_pool=2,
        **kwargs,
    )


class MAEHiera(LightningModule):
    default_backbone = "mae_hiera_base_plus_112"

    def __init__(
        self,
        backbone: str,
        batch_size_per_device: int,
        in_channels: int,
        img_size: int,
        num_classes: int,
        has_online_classifier: bool,
        train_transform: Module,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        last_backbone_channel: int = None,
    ):
        assert (
            last_backbone_channel is None
        ), f"change of last backbone channel is not supported (given: {last_backbone_channel})"

        super().__init__()
        self.save_hyperparameters(ignore=["train_transform"])
        self.hparams["method"] = self.__class__.__name__
        self.batch_size_per_device = batch_size_per_device
        if backbone == "default":
            backbone = self.default_backbone
            print_rank_zero(f"Using default backbone: {backbone}")

        if backbone in sys.modules[__name__].__dict__:
            model_cls = sys.modules[__name__].__dict__[backbone]
        elif backbone in hiera.__dict__:
            model_cls = hiera.__dict__[backbone]
        else:
            raise ValueError(f"Backbone {backbone} not found")

        self.model: MaskedAutoencoderHiera = model_cls(in_chans=in_channels)
        self.model.head = Identity()
        self.model.norm = Identity()

        if img_size != self.model.tokens_spatial_shape[0] * self.model.patch_stride[0]:
            raise ValueError(
                f"Image size {img_size} does not match model resolution "
                f"{self.model.tokens_spatial_shape[0] * self.model.patch_stride[0]}"
            )

        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.feature_dim = self.model.encoder_norm.weight.shape[0]
        self.last_backbone_channel = self.feature_dim
        self.norm_pix_loss = norm_pix_loss

        self.has_online_classifier = has_online_classifier
        if has_online_classifier:
            self.online_classifier = OnlineLinearClassifier(
                feature_dim=self.feature_dim, num_classes=num_classes
            )

        self.train_transform = train_transform

    def forward(self, x: Tensor) -> Tensor:
        # ensuring that the img size requirements are met (this should only be triggered for offline eval)
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(x, self.img_size)
        
        _, intermediates = super(MaskedAutoencoderHiera, self.model).forward(
            x, return_intermediates=True
        )
        features = intermediates[-1].mean(dim=(1, 2))
        return features


    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = self.model.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super(MaskedAutoencoderHiera, self.model).forward(
            x, mask, return_intermediates=True
        )

        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.model.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        for head, interm_x in zip(self.model.multi_scale_fusion_heads, intermediates):
            x += hiera_mae.apply_fusion_head(head, interm_x)

        x = self.model.encoder_norm(x)

        last_intermediate = intermediates[-1]
        return x, mask, last_intermediate

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.model.q_stride) == 2:
            label = self.model.get_pixel_label_2d(x, mask, norm=norm)
        elif len(self.model.q_stride) == 3:
            label = self.model.get_pixel_label_3d(x, mask, norm=norm)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label
    
    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        images = batch[0]
        # Create views
        with torch.no_grad():
            images = self.train_transform(images)[0]  # only expecting single view

        latent, mask, last_intermediate = self.forward_encoder(images, self.mask_ratio)
        pred, pred_mask = self.model.forward_decoder(
            latent, mask
        )  # pred_mask is mask at resolution of *prediction*

        loss, _, _ = self.forward_loss(images, pred, ~pred_mask, norm=self.norm_pix_loss)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(images))

        # Online linear evaluation.
        if self.has_online_classifier:
            targets = batch[1]
            features = last_intermediate.mean(dim=(1, 2, 3)) # pooling over spatial dimensions
            cls_loss, cls_log = self.online_classifier.training_step(
                (features.detach(), targets), batch_idx
            )
            self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
            loss = loss + cls_loss
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        images = batch[0]
        if self.has_online_classifier:
            # ensuring that the img size requirements are met
            if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
                images = F.interpolate(images, self.img_size)

            targets = batch[1]
            cls_features = self.forward(images)
            cls_loss, cls_log = self.online_classifier.validation_step(
                (cls_features.detach(), targets), batch_idx
            )
            self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
            return cls_loss
        else:
            return None  # Could return variance and covariance loss instead

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = utils.get_weight_decay_parameters([self.model])
        param_list = [
            {"name": "mae", "params": params},
            {
                "name": "mae_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },
        ]
        if self.has_online_classifier:
            param_list.append(
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                }
            )
        optimizer = AdamW(
            param_list,
            lr=1.5e-4 * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches / self.trainer.max_epochs * 40
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

