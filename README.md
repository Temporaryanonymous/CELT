# CELT: Using feature layer interactions to improve semantic segmentation models

<p float="center">
  <img width="185" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/MANet.gif"/>
  &nbsp;
  &nbsp;
  <img width="185" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Unet.gif"/> 
    &nbsp;
    &nbsp;
  <img width="170" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/PSPnet.gif"/>
    &nbsp;
    &nbsp;
  <img width="205" height="175" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/FeaturePN.gif"/>
    &nbsp;
    &nbsp;
  <img width="185" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Unet%2B%2B.gif"/>
     &nbsp;
    &nbsp;
  <img width="190" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/DeepLabV3.gif"/>
     &nbsp;
    &nbsp;
  <img width="180" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Linknet.gif"/>
     &nbsp;
    &nbsp;
  <img width="170" height="200" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/PAN.gif"/>
</p>

<p align="center">
  <img width="500" height="70" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Figure%20Legend.jpg">
</p>

This is the source code for the method as described in our paper:
**CELT: Using feature layer interactions to improve semantic segmentation models**. The lines 81-139 of Architecture/encoder/resnet.py are about how to insert CELT into the encoder of the existing segmentation models. You can apply CELT on your own models, which is a very easy idea to implement.

## Data

In order to make it easier for the readers to reproduce and understand the code, we have provided a small amount of example data used in our experiment under the **image_samples** folder, where each dataset (CamVid, Skin Lesion, CUB Birds) provides five training, validation and test images.


## File declaration


**Architecture/encoder**：contains the encoding part of all eight CELT plugged models.

**Architecture/manet**：the decoder and segmentation part of the manet_celt model.

**Architecture/unet**： the decoder and segmentation part of the unet_celt model.

**Architecture/unetplusplus**：the decoder and segmentation part of the unet++_celt model.

**Architecture/deeplabv3**：the decoder and segmentation part of the deeplabv3_celt model.

**Architecture/fpn**：the decoder and segmentation part of the fpn_celt model.  

**Architecture/pan**：the decoder and segmentation part of the pan_celt model.

**Architecture/linknet**：the decoder and segmentation part of the linknet_celt model.

**Architecture/pspnet**：the decoder and segmentation part of the pspnet_celt model.

**CELT_Unet_CamVid.py**: The Unet model with CELT method, which is used for CamVid dataset. You can change this file to test other models and datasets.

## Run the codes
Install the environment.
```bash
pip install -r requirements.txt
```

Train and test the model.
```bash
python CELT_Unet_CamVid.py
```
