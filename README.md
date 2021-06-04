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
**CELT: Using feature layer interactions to improve semantic segmentation models**. The lines 81-139 of Architecture/encoder/resnet.py are about how to insert CELT into the encoder of the existing segmentation models. You can apply CELT on your own models, which is a very easy idea to method.

## Data

In order to use the code, you have to provide 
* [Theraputic Target Database](http://db.idrblab.net/ttd/full-data-download)  You don't need to download by yourself, I have uploaded all the TTD 2016 version in *<./data/TTD>*. 
* [SemedDB](https://skr3.nlm.nih.gov/SemMedDB/) **You need to download from [here](https://pan.baidu.com/s/1zuOELNGAua6i523_nLK6mw)** with password:1234 to obtain the whole knowledge graph. After downloading the "predications.txt" file, please replace the file *<./data/SemedDB/predications.txt>*. with this **new** downloaded file. 

## Run the codes
Install the environment.
```bash
pip install -r requirements.txt
```

Construct training and test data.
```bash
python experimental_data.py
```

Train and test the model.
```bash
python main.py
```

## Illustration of feature selection
<div align=center><img width="800" height="300" src="https://github.com/ShengtianSang/SemaTyP/blob/main/figures/Illustration_of_Feature_selection.jpg"/></div>
<p align="center">
An illustration of the features constructed in our work.
</p>


## File declaration

**data/SemmedDB**： contains all relations extracted from SemmedDB, which are used for constructing the Knowledge Graph in our experiment. The whole "predications.txt" contains **39,133,975** relations, we just leave a small sample "predications.txt" file here which contain **100** relation. The whole "predications.txt" file coule be downloaded from 
  
**data/TTD**： contains the drug, target and disease relations retrieved from Theraputic Target Database.
    
**experimental_data.py**: constuct the drug-target-disease associations from TTD and Knowledge Graph.

**knowledge_graph.py**: construct the Knowledge Graph used in our experiment.
 
**data_loader.py**：used to load traing and test data.

**main.py**：used to train and test the models


