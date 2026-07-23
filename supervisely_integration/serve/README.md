<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/STDC-Seg/releases/download/v0.0.1/poster.jpg"/>

# Serve STDC-Seg

Deploy [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg) semantic segmentation models in Supervisely.

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#supported-models">Supported Models</a> •
  <a href="#how-to-run">How to Run</a> •
  <a href="#external-models">External Models</a> •
  <a href="#model-configuration">Model Configuration</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/stdc-seg/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/STDC-Seg)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/stdc-seg/supervisely_integration/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/stdc-seg/supervisely_integration/serve.png)](https://supervisely.com)

</div>

## Overview

Serve STDC-Seg models through the Supervisely inference API and use them in
image and video semantic segmentation workflows.

The application supports:

- publicly available pretrained STDC1-Seg and STDC2-Seg checkpoints
- external PyTorch checkpoints trained outside Supervisely
- CPU and GPU inference
- class names and colors supplied through a YAML model configuration

Custom checkpoints trained inside Supervisely and ONNX/TensorRT runtimes are
not supported.

## Supported Models

| Model | Backbone | Dataset | Input size (W × H) |
| --- | --- | --- | --- |
| STDC1-Seg50 | STDCNet813 | Cityscapes | 1024 × 512 |
| STDC1-Seg75 | STDCNet813 | Cityscapes | 1536 × 768 |
| STDC2-Seg50 | STDCNet1446 | Cityscapes | 1024 × 512 |
| STDC2-Seg75 | STDCNet1446 | Cityscapes | 1536 × 768 |

The `input_size` arrays in
[`supervisely_integration/models.json`](../models.json) use `[width, height]`
order.

## How to Run

1. Run **Serve STDC-Seg** from the Supervisely Ecosystem.
2. Open the **Publicly available models** tab.
3. Select a pretrained model.
4. Select a device.
5. Click **Serve** and wait for deployment to finish.

After deployment, the application displays the selected backbone, input size,
inference settings, and the 19 Cityscapes output classes. The running session
can then be selected by Supervisely inference applications.

## External Models

Use the **External Models** tab to deploy a compatible STDC-Seg checkpoint that
was trained outside Supervisely.

Prepare the following files in Supervisely Team Files:

- a PyTorch checkpoint (`.pth`);
- a YAML model configuration.

Select both files in the application, choose a device, and click **Serve**.

The application determines `STDCNet813` or `STDCNet1446` from the checkpoint
structure. The input size, class names, and optional class colors are read from
the YAML configuration. A separate classes JSON file is not required.

The number of entries in `SEG_CATEGORIES` must match the number of output
channels in the checkpoint.

## Model Configuration

An external model configuration must define:

- `NET_INP_WIDTH` — inference image width;
- `NET_INP_HEIGHT` — inference image height;
- `SEG_CATEGORIES` — mapping of class names to contiguous class indices,
  starting at `0`.

`SEG_COLORS` is optional. When omitted, the application generates a color
palette automatically.

Example configuration for the pretrained STDC2-Seg75 Cityscapes checkpoint:

```yaml
NET_INP_WIDTH: 1536
NET_INP_HEIGHT: 768

SEG_CATEGORIES:
  road: 0
  sidewalk: 1
  building: 2
  wall: 3
  fence: 4
  pole: 5
  traffic light: 6
  traffic sign: 7
  vegetation: 8
  terrain: 9
  sky: 10
  person: 11
  rider: 12
  car: 13
  truck: 14
  bus: 15
  train: 16
  motorcycle: 17
  bicycle: 18

SEG_COLORS:
  0: [128, 64, 128]
  1: [244, 35, 232]
  2: [70, 70, 70]
  3: [102, 102, 156]
  4: [190, 153, 153]
  5: [153, 153, 153]
  6: [250, 170, 30]
  7: [220, 220, 0]
  8: [107, 142, 35]
  9: [152, 251, 152]
  10: [70, 130, 180]
  11: [220, 20, 60]
  12: [255, 0, 0]
  13: [0, 0, 142]
  14: [0, 0, 70]
  15: [0, 60, 100]
  16: [0, 80, 100]
  17: [0, 0, 230]
  18: [119, 11, 32]
```

The default Cityscapes configuration is available in
[`supervisely_integration/model_config.yaml`](../model_config.yaml).

Class names and colors are serving metadata; standard STDC-Seg checkpoints only
contain model parameters and the number of output channels. The application
creates a Supervisely `ProjectMeta` from the YAML configuration during
deployment.

## Acknowledgment

This application is based on
[STDC-Seg](https://github.com/MichaelFan01/STDC-Seg) and the paper
[Rethinking BiSeNet for Real-time Semantic Segmentation](https://arxiv.org/abs/2104.13188).
