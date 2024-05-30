# 2024Sprint Fudan DIPpj: DM2F-Net

## Installation & Preparation

Make sure you have `Python>=3.7` installed on your machine.

**Environment setup:**

1. Create conda environment

       conda create -n dm2f
       conda activate dm2f

2. Install dependencies (test with PyTorch 1.8.0):

   1. Install pytorch==1.8.0 torchvision==0.9.0 (via conda, recommend).

   2. Install other dependencies

          pip install -r requirements.txt

* Prepare the dataset

   * Download the RESIDE dataset from the [official webpage](https://sites.google.com/site/boyilics/website-builder/reside).

   * Download the O-Haze dataset from the [official webpage](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

   * Make a directory `./data` and create a symbolic link for uncompressed data, e.g., `./data/RESIDE`.

## Training

1. Set the path of datasets in tools/config.py
2. Run DM2FNet in RESIDE by ```python train.py```
3. Run DM2FNet in O-HAZE by ```python train_ohaze.py```
4. Run improved algorithm in RESIDE by ```python train_new.py```
5. Run improved algorithm in O-HAZE by ```python train_ohaze_new.py```

~~The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.~~

Use pretrained ResNeXt (resnext101_32x8d) from torchvision.

*Hyper-parameters* of training were set at the top of *train.py*, and you can conveniently
change them as you need.

## Testing

1. Set the path of five benchmark datasets in tools/config.py.
2. Put the trained model in `./ckpt/`.
2. Run DM2FNetby ```python test.py```
4. Run improved algorithm in RESIDE by ```python test_new.py```
5. Run in the dataset collected by myself by ```python mytest.py```

*Settings* of testing were set at the top of `test.py`, and you can conveniently
change them as you need.


