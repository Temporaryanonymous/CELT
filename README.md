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
**CELT: Using feature layer interactions to improve semantic segmentation models**. The lines 81-139 of Architecture/encoders/resnet.py are about how to insert CELT into the encoder of the existing segmentation models. You can apply CELT on your own models, which is a simple and very easy to implement idea.

## Requirements
* scikit-learn
* numpy
* tqdm

## Data


## Run the codes
Install the environment.
```bash
pip install -r requirements.txt
```

