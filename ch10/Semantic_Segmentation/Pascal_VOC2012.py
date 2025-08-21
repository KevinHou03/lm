import os
import torch
import torchvision
from LM.d2l import DATA_HUB, DATA_URL,download_extract

DATA_HUB['voc2012'] = (DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = download_extract('voc2012', 'VOCdevkit/VOC2012')
print(voc_dir)

