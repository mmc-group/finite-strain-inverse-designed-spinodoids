from core.__importList__ import *
import torch
import sys 
import numpy as np
import os

np.random.seed(123)
torch.manual_seed(123)
drivers_path = os.getcwd()
project_path = os.path.dirname(drivers_path)
data_path = 'data/'
directions = ['X','Y','Z']

n_hidden_lam=[16,16,16]
n_hidden_col=[32,32,32]
n_hidden_cub=[48,48,48]
bv_hid=[3,3,3]
n_output=1
dim_y=1
traintest_split=0.1
numWorkers=1
batchSize=512
opt_epochs=30
device = 'cpu'
epochs=3000
lr=0.005
opt_lr=0.1
time_steps=80
max_eps=0.39
query_factor=1.2

def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div

def lossFn(y, y_hat, mask=None):
    mask = torch.ones_like(y)
    mask = divide_no_nan(mask, torch.abs(y))
    mape = torch.abs(y - y_hat) * mask
    mape = torch.mean(mape)
    
    return mape


def get_direction_case(index):
    """
    Retrieve the direction case string based on the given index.

    Args:
        index (int): The index corresponding to the direction case.

    Returns:
        str: The direction case string (e.g., 'lamX', 'colY', etc.).

    Raises:
        ValueError: If the index is not in the valid range.
    """
    direction_case_dict = {
        0: "lamX",
        1: "colX",
        2: "cubX",
        3: "lamY",
        4: "colY",
        5: "cubY",
        6: "lamZ",
        7: "colZ",
        8: "cubZ"
    }
    
    if index not in direction_case_dict:
        raise ValueError(f"Invalid index {index}. Valid indices are 0 to 8.")

    return direction_case_dict[index]