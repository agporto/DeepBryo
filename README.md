<p align="center">
<img src="resources/logo_red.png" alt="DeepBryo logo" width='200' height='200' >
</p>

# AI-assisted segmentation of cheilostome bryozoans


<p align="center">
  <img src="resources/deepbryo.gif" alt="animated" />
</p>

This repo contains the code and configuration files necessary to initiate a local instance of `DeepBryo`. It is based on [mmdetection](https://github.com/open-mmlab/mmdetection), [streamlit](https://streamlit.io/) and [SwinTransformer](https://arxiv.org/pdf/2103.14030.pdf)

## Server 

We host a `DeepBryo` production server for bryozoologists. It can be found at the [BryoLab](https://bryolab.ngrok.io). Please complete [this](https://docs.google.com/forms/d/e/1FAIpQLSc-NoKamdaWiB9pCGQyXFHsMpXXlBgYRlwwSn53h8jwf7UMnw/viewform?usp=pp_url) registration form to get access. 


## Updates

07/01/2022 - `DeepBryo` v0.1 gets an official Github repository.

## Usage

Model weights will be made available upon publication.

Once the installation procedures are complete, please download the [model weights]() and save the file `deepbryo.pth` inside the `inference/` folder. After that, you can launch a `DeepBryo` server using the following command:
```
streamlit run app/app.py --server.port 8080

```

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.


### Training

To retrain `DeepBryo`, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```


**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


## Citing DeepBryo
```
To be announced 

```

## Other Links



