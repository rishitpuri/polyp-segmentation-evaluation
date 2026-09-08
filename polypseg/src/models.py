"""Model factory.

All three architectures come from segmentation_models_pytorch so that the
encoder, pretraining source and training loop are identical across models —
the architecture is the only variable in the comparison.
"""
import segmentation_models_pytorch as smp

# encoder_weights="imagenet" for every model keeps pretraining comparable.
#
# The set spans four backbone families and an order of magnitude in capacity
# (3.7M-28M), so conclusions are not tied to one architectural style:
#   - classic CNN encoder-decoder (U-Net)
#   - densely nested CNN (U-Net++)
#   - hierarchical transformer, small and mid-size (SegFormer-B0/B2)
#   - pyramid vision transformer backbone, the family used by recent
#     specialised polyp networks such as Polyp-PVT (U-Net + PVTv2-B2)
MODELS = {
    "unet": dict(arch="Unet", encoder_name="resnet34"),
    "unetpp": dict(arch="UnetPlusPlus", encoder_name="resnet34"),
    "segformer": dict(arch="Segformer", encoder_name="mit_b0"),
    "segformer_b2": dict(arch="Segformer", encoder_name="mit_b2"),
    "unet_pvt": dict(arch="Unet", encoder_name="tu-pvt_v2_b2"),
}


def build_model(name: str):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    cfg = dict(MODELS[name])
    arch = cfg.pop("arch")
    return smp.create_model(
        arch=arch,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        **cfg,
    )


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
