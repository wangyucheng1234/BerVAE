# BerVAE
This repo holds the Pytorch implementation of BerVAE. Our implementation is based on [BTH]. 

## Environment
```
Python 3.7.0
```

## Requirement
```
scipy==1.6.3
h5py==3.1.0
torch==1.8.1+cu111
matplotlib==3.4.2
pandas==1.2.4
numpy==1.19.5
tensorboardX==2.5.1
```

## Quick Start

### Download Features

VGG features of FCV are kindly uploaded by the authors of [SSVH]. You can download them from [Baiduyun] disk.

Please set the data_root and home_root in ```args.py``` in both ```./utils/``` and ```./model/```. 
You can place these features to in data_root.

### Training BerVAE
After correctly setting the path, you can run train.py to train the model. Models will be saved in ```./models. ```

### Testing BerVAE
When training is done, you can run ```eval.py``` to test it. mAP files will be save in ```./results.``` The scatter and box-plots of the AP$@K$ values of the query video clips with respect to the uncertainty level of corresponding hash-codes of the queries will also be saved in ```./```
To evaluate the uncertainty quantification performance using our proposed IDU, you can run ```eval_uq.py```.



[SSVH]:https://github.com/lixiangpengcs/Self-Supervised-Video-Hashing
[BTH]:https://github.com/Lily1994/BTH
[Baiduyun]:https://pan.baidu.com/s/1i65ccHv