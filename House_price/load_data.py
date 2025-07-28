import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

train_data = pd.read_csv('/Users/kevinhou/Documents/PyTorch_Datasets/kaggle_house_price/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/Users/kevinhou/Documents/PyTorch_Datasets/kaggle_house_price/house-prices-advanced-regression-techniques/test.csv')
