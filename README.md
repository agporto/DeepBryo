<p align="center">
<img src="resources/logo_red.png" alt="DeepBryo logo" width='200' height='200' >
</p>

# AI-assisted segmentation of cheilostome bryozoans


<p align="center">
  <img src="resources/deepbryo.gif" alt="animated" />
</p>

This repo contains the code and configuration files necessary to initiate a local instance of `DeepBryo`. It is based on [mmdetection](https://github.com/open-mmlab/mmdetection), [streamlit](https://streamlit.io/) and [SwinTransformer](https://arxiv.org/pdf/2103.14030.pdf)

## Server 

We host a `DeepBryo` production server for bryozoologists. It can be found at [DeepBryo](https://deepbryo.ngrok.io). Please complete [this](https://docs.google.com/forms/d/e/1FAIpQLSc-NoKamdaWiB9pCGQyXFHsMpXXlBgYRlwwSn53h8jwf7UMnw/viewform?usp=pp_url) registration form to let us know who you are and what is your main goal when using it.


## Updates
11/18/2022 - Preprint is out at bioRxiv. Model weights are released.

11/04/2022 - CLI-version of the app released

07/01/2022 - `DeepBryo` v0.1 gets an official Github repository.

## Usage

Once the installation procedures are complete, please download the [model weights](https://drive.google.com/file/d/1vKf7joCt_QNwwq4IFauWYA4Lh8FMcBPo/view?usp=share_link) and save the file `deepbryo-tiny.pth` inside the `inference/` folder. After that, you can launch a `DeepBryo` server using the following command:

```
streamlit run app/app.py --server.port 8080
```
The web app can then be launched on a web browser at: 
```
localhost:8080
```
Note that you are free to choose other server ports. Also, you can serve the app over the internet by forwarding the port in question to your own domain.

**Note:** `Windows` users having trouble with streamlit, please see [this](https://discuss.streamlit.io/t/getting-an-error-streamlit-is-not-recognized-as-an-internal-or-external-command-operable-program-or-batch-file/361).

## Installation

**Note:** As of now, macOS is not supported by DeepBryo due to lack of CUDA support. 


Below are quick steps for installation using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) in `Linux` or `Windows` (assuming the presence of an `NVIDIA` gpu):

```
conda create -n deepbryo python=3.7 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11 -c pytorch -y
conda activate deepbryo
pip3 install openmim
mim install mmcv-full==1.4.0
git clone https://github.com/agporto/DeepBryo.git
cd DeepBryo
pip3 install -e .
```


## High-Throughput Inference (Command-line Interface)

If you would rather use the model as a command-line tool to perform high-throughput prediction. Simply use the following command:

```
python app/app-cli.py -i INPUT_DIR -o OUT_DIR [other optional arguments]
```

The parameters associated with the cli tool mirror the web app and are:

```
usage: app-cli.py [-h] -i INPUT_DIR -o OUT_DIR [-c CLASS] [-p PADDING [PADDING ...]] [-t CONFIDENCE] 
                  [-a] [-s STRICTNESS] [-sc SCALE]

optional arguments:
  -h, --help            show this help message and exit

  -i INPUT_DIR, --input_dir INPUT_DIR
                        folder containing images to be predicted (required)

  -o OUT_DIR, --out-dir OUT_DIR
                        output folder (required)

  -c CLASS, --class CLASS
                        object class of interest. 
                        options: all, autozooid, orifice, avicularium, ovicell, ascopore, opesia

  -p PADDING [PADDING ...], --padding PADDING [PADDING ...]
                        remove objects falling within a certain distance from
                        the image border. please provide it as a list in the
                        following order: left, top, right, bottom

  -t CONFIDENCE, --confidence CONFIDENCE
                        model's confidence threshold (default = 0.5)

  -a, --autofilter      enable autofilter of model predictions

  -s STRICTNESS, --strictness STRICTNESS
                        regulated the strictness of the automated filtering algorithm

  -sc SCALE, --scale SCALE
                        pixel-to-um scaling parameter (default = None)
```




## Training

To retrain `DeepBryo`, run:
```bash
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
**Note:** For other details, please see the [SwinTransformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) official web page.  

## Testing

To test a `DeepBryo` model checkpoint, please use: 

```bash
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```
## Customizing DeepBryo to other study systems

The `DeepBryo` repository is designed to work with the `COCO JSON` format, which is a common format for object detection and segmentation tasks. If you want to adapt this repository to work with other study systems, you would need to prepare your data in the `COCO JSON` format and make some modifications to the code. Here are are some key steps:

* __Prepare Your Data__: First, you need to annotate your images in the `COCO JSON` format. This format includes information about the image size, the objects present in the image, their categories, and their bounding box coordinates or segmentation masks. There are many tools available online that can help you annotate your images in this format, such as Labelbox, VGG Image Annotator (VIA), CVAT, and others.

* __Modify Configuration Files__: The `DeepBryo` repository contains configuration files (found in the configs directory) that specify the model architecture and training parameters. You would need to modify these files to suit your specific task. For example, you might need to change the number of classes if your task involves different categories of objects. You will also need to give the data paths containing the train/test data.

* __Update Training and Testing Scripts__: The training and testing scripts (found in the tools directory) may also need to be updated. For instance, you might need to adjust the evaluation metrics if your task requires different measures of performance.

* __Retrain the Model__: Once you've made these modifications, you can retrain the model on your data. The commands for training the model are provided above. You might need to adjust these commands depending on your specific setup (e.g., if you're using multiple GPUs).

* __Test the Model__: After training, you can test the model on your data. Again, the commands for testing the model are provided above.

* __Adjust the User Interface__: The app.py file in the app directory contains the code for the Streamlit user interface. You might need to adjust this to suit your specific task. For example, you might need to change the path to the model checkpoints and the config files. You will also need to edit the _classes_ variable and the logo (should you want).

Remember, adapting a model to a new task can be a complex process that requires a good understanding of the model architecture and the specific requirements of your task. It might take some trial and error to get everything working correctly.

## Citing DeepBryo
```bibtex
@article {DiMartino_DeepBryo,
	author = {Di Martino, Emanuela and Berning, Bjorn and Gordon, Dennis P. and Kuklinski, Piotr and Liow, Lee Hsiang and Ramsfjell, Mali H. and Ribeiro, Henrique L. and Smith, Abigail M. and Taylor, Paul D. and Voje, Kjetil L. and Waeschenbach, Andrea and Porto, Arthur},
	title = {DeepBryo: a web app for AI-assisted morphometric characterization of cheilostome bryozoans},
	year = {2022},
	doi = {10.1101/2022.11.17.516938},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/11/17/2022.11.17.516938},
	eprint = {https://www.biorxiv.org/content/early/2022/11/17/2022.11.17.516938.full.pdf},
	journal = {bioRxiv}
}
```

## Other Links

[BRYOZOA.NET](http://bryozoa.net/): Great resource for all things Bryozoa and home of [IBA](http://bryozoa.net/iba/).

[WORMS](https://www.marinespecies.org/): World register of marine species. `DeepBryo`'s taxonomy follows `Worms`.


