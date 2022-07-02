<p align="center">
<img src="resources/logo_red.png" alt="DeepBryo logo" width='200' height='200' >
</p>

## A web app for AI-assisted segmentation of cheilostome bryozoan colonies

<p align="center">
  <img src="resources/deepbryo.gif" alt="animated" />
</p>

This repo contains the supported code and configuration files necessary to initiate a local instance of [DeepBryo](). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [SwinTransformer](https://arxiv.org/pdf/2103.14030.pdf)

## Demo 

A demo version of DeepBryo can be found at the [BryoLab]('https://bryolab.ngrok.io')

## Updates

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```

```

### Training

To retrain DeepBryo, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```


**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


## Citing DeepBryo
```

```

## Other Links



