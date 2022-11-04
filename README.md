<p align="center">
<img src="resources/logo_red.png" alt="DeepBryo logo" width='200' height='200' >
</p>

# AI-assisted segmentation of cheilostome bryozoans


<p align="center">
  <img src="resources/deepbryo.gif" alt="animated" />
</p>

This repo contains the code and configuration files necessary to initiate a local instance of `DeepBryo`. It is based on [mmdetection](https://github.com/open-mmlab/mmdetection), [streamlit](https://streamlit.io/) and [SwinTransformer](https://arxiv.org/pdf/2103.14030.pdf)

## Server 

We host a `DeepBryo` production server for bryozoologists. It can be found at the [BryoLab](https://bryolab.ngrok.io). Please complete [this](https://docs.google.com/forms/d/e/1FAIpQLSc-NoKamdaWiB9pCGQyXFHsMpXXlBgYRlwwSn53h8jwf7UMnw/viewform?usp=pp_url) registration to let us know who you are and why you are using it. It will help us convincing funders of the project's importance.


## Updates

07/01/2022 - `DeepBryo` v0.1 gets an official Github repository.

## Usage

Model weights will be made available upon publication.

Once the installation procedures are complete, please download the [model weights]() and save the file `deepbryo.pth` inside the `inference/` folder. After that, you can launch a `DeepBryo` server using the following command:

```
streamlit run app/app.py --server.port 8080

```

### Installation

Below are quick steps for installation:

```
conda create -n deepbryo python=3.7 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11 -c pytorch -y
conda activate deepbryo
pip3 install openmim
mim install mmcv-full==1.4.0
git clone https://github.com/agporto/DeepBryo.git
cd DeepBryo
pip3 install -e .

```


### Inference



### Training

To retrain `DeepBryo`, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 

```
**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.

### Testing
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm

```

## Citing DeepBryo
```
To be announced 

```

## Other Links

[Bryozoa.net](http://bryozoa.net/): Great resource for all things Bryozoa and home of [IBA](http://bryozoa.net/iba/).
[WORMS](https://www.marinespecies.org/): World register of marine species. `DeepBryo`'s taxonomy follows `Worms`.


