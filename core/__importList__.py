from .normalization import Normalization
from os.path import isfile, join
import os
import pathlib
from torch.utils.data import DataLoader

#torch
import torch
from torch.utils.data import TensorDataset

#torch options
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

#numpy
import numpy as np
from numpy import inf

#matplotlib
import matplotlib.pyplot as plt

#scipy
import scipy
from scipy import interpolate

#pandas
import pandas as pd

#others:
import os
from os.path import join, isfile
import sys
from pickle import dump, load
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm