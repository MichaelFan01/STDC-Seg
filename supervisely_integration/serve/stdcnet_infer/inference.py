from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .model import BiSeNet

_to_tensor = transforms.Compose([transforms.ToTensor()])

def load_model_config(path):
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must contain a YAML mapping: {path}")
    return config


def _build_palette(n_classes):
    """
    Generate a distinct RGB color per class (class 0 = black background).
    """
    hues = np.linspace(0, 179, n_classes, dtype=np.uint8)
    hsv = np.stack([hues, np.full_like(hues, 200), np.full_like(hues, 230)], axis=1)
    rgb = cv2.cvtColor(hsv[None, :, :], cv2.COLOR_HSV2RGB)[0]
    rgb[0] = [0, 0, 0]
    return rgb.astype(np.uint8)

class STDCNetSegmenter:
    """
    Loads STDCNet weights once and runs semantic segmentation on images.
    """

    BACKBONE_FROM_CHECKPOINT = "from_checkpoint"

    def __init__(self, weights_path, device="cpu", n_classes=None,
                 backbone=None, input_size=None, config_path=None):
        self.device = torch.device(device)

        # Find a model_config.yaml next to the weights unless one is given.
        cfg = None
        if config_path is None:
            guess = Path(weights_path).parent / "model_config.yaml"
            config_path = guess if guess.exists() else None
        if config_path is not None:
            cfg = load_model_config(config_path)
        cfg = cfg or {}
        state_dict = self._extract_state_dict(weights_path)
        if backbone == self.BACKBONE_FROM_CHECKPOINT:
            backbone = self._infer_backbone(state_dict)
        elif backbone is None:
            raise ValueError(
                "Backbone is not provided. Pass a concrete backbone or "
                "STDCNetSegmenter.BACKBONE_FROM_CHECKPOINT."
            )
        if backbone not in ("STDCNet813", "STDCNet1446"):
            raise ValueError(f"Unsupported STDC backbone: {backbone}")
        self.backbone = backbone
        self.n_classes = (
            n_classes
            or self._infer_n_classes(state_dict)
            or (len(cfg["SEG_CATEGORIES"]) if "SEG_CATEGORIES" in cfg else None)
        )
        if self.n_classes is None:
            raise ValueError(
                "Could not determine n_classes from weights or config; "
                "pass n_classes explicitly."
            )
        if "SEG_CATEGORIES" in cfg:
            self.class_names = [
                name for name, _ in sorted(cfg["SEG_CATEGORIES"].items(),
                                           key=lambda kv: kv[1])
            ]
        else:
            self.class_names = [f"class_{i}" for i in range(self.n_classes)]
        if len(self.class_names) != self.n_classes:
            raise ValueError(
                "Config and checkpoint have different numbers of classes: "
                f"{len(self.class_names)} and {self.n_classes}."
            )
        if input_size is not None:
            self.input_size = tuple(input_size)
        elif "NET_INP_WIDTH" in cfg and "NET_INP_HEIGHT" in cfg:
            self.input_size = (int(cfg["NET_INP_WIDTH"]), int(cfg["NET_INP_HEIGHT"]))
        else:
            raise ValueError(
                "Input size is not provided. Pass input_size explicitly or define "
                "NET_INP_WIDTH and NET_INP_HEIGHT in the model config."
            )
        self.palette = self.build_palette(cfg, self.n_classes)

        # --- build & load ---
        self.model = BiSeNet(
            backbone=backbone, n_classes=self.n_classes,
            use_boundary_2=False, use_boundary_4=False,
            use_boundary_8=True, use_boundary_16=False, use_conv_last=False,
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        bad_missing = [k for k in missing if "num_batches_tracked" not in k]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint doesn't match model.\n"
                f"  missing: {bad_missing}\n  unexpected: {unexpected}"
            )

        self.model.to(self.device).eval()

    @staticmethod
    def _extract_state_dict(weights_path):
        """
        Return the raw {param: tensor} dict from either a wrapped training
        checkpoint ({'model': ...} / {'state_dict': ...}) or a plain state_dict.
        """
        ckpt = torch.load(weights_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
        return ckpt
    
    @staticmethod
    def _infer_n_classes(state_dict):
        w = state_dict.get("conv_out.conv_out.weight")
        return int(w.shape[0]) if w is not None else None

    @staticmethod
    def _infer_backbone(state_dict):
        marker = "cp.backbone.features."
        feature_indices = {
            int(key.split(marker, 1)[1].split(".", 1)[0])
            for key in state_dict
            if marker in key
        }
        max_feature_index = max(feature_indices, default=None)
        if max_feature_index == 7:
            return "STDCNet813"
        if max_feature_index == 13:
            return "STDCNet1446"
        raise ValueError(
            "Could not determine STDC backbone from checkpoint. "
            f"Expected the last feature index to be 7 or 13, got {max_feature_index}."
        )

    @staticmethod
    def build_palette(cfg, n_classes):
        if "SEG_COLORS" in cfg:
            palette = _build_palette(n_classes)  # start from auto, override provided
            for idx, rgb in cfg["SEG_COLORS"].items():
                palette[int(idx)] = np.array(rgb, dtype=np.uint8)
            return palette
        return _build_palette(n_classes)

    @staticmethod
    def _read_image(image):

        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected an HxWx3 image array.")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)  # assume BGR in, like cv2.imread
    
    @torch.no_grad()
    def predict(self, image, overlay=False):
        """
        Run segmentation.

        image   : path or HxWx3 BGR array
        overlay : if True, return an RGB uint8 image with the mask blended on top;
                  otherwise return the (H, W) uint8 class-index map.
        """
        image_rgb = self._read_image(image)
        resized = cv2.resize(image_rgb, self.input_size)  # (W, H)

        x = _to_tensor(resized).unsqueeze(0).to(self.device)  # 1x3xHxW, scaled to [0,1]
        logits = self.model(x)[0]                              # main seg head
        logits = F.interpolate(logits, size=image_rgb.shape[:2],
                               mode="bilinear", align_corners=True)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        preds = preds.squeeze(0).to(torch.uint8).cpu().numpy()

        return self.overlay(image_rgb, preds) if overlay else preds

    def overlay(self, image_rgb, preds, alpha=0.45):
        color_mask = self.palette[preds]
        return cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)
