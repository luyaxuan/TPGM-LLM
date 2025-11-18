import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import time  # Already imported, no need to repeat
import util
import os
from util import *
import random
from model_bert_GCN import TPGM_LLM
from ranger21 import Ranger
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="/root/autodl-tmp/s_t_1_shuffled", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--input_text_dim", type=int, default=768, help="text embedding feature dimension")# Text embedding feature dimension
parser.add_argument("--channels", type=int, default=64, help="number of features")
parser.add_argument("--num_nodes", type=int, default=1296, help="number of nodes")
parser.add_argument("--input_len", type=int, default=15, help="input_len")
parser.add_argument("--output_len", type=int, default=15, help="out_len")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=50, help="200~500")
parser.add_argument("--print_every", type=int, default=100, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)
args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
path = args.save + args.data + "/"

print(args)

model = TPGM_LLM(device, 
    args.input_dim, 
    args.channels, 
    args.num_nodes, 
    args.input_len, 
    args.output_len, 
    args.dropout)
model.to(device)
best_path='/root/TPGM-LLM-main/logs/2025-11-05-16:34:35-/root/autodl-tmp/s_t_1_shuffled/best_model.pth'
model.load_state_dict(torch.load(best_path), strict=False)
model.eval()

# Normalization
def normalize(x, mean, std):
    return (x - mean) / std

# Denormalization
def denormalize(x, mean, std):
    return x * std + mean

# Normalization parameters: these parameters are calculated based on training data
mean =  53.44957185879243
std = 19.08435640827557

# Sensor data loading and preprocessing
sensor_data = np.load('/root/TPGM-LLM-main/new_npy/sensor_input.npy')
sensor_data = sensor_data[..., 1:]  # Slice processing
sensor_tensor = torch.Tensor(sensor_data).to(device)
sensor_tensor = sensor_tensor.transpose(1, 3)  # Adjust shape

# Text data loading and preprocessing
text_data = np.load('/root/TPGM-LLM-main/new_npy/text.npy')
text_tensor = text_data  # Assumed to be preprocessed into a format directly usable by the model

# Inference time measurement
with torch.no_grad():
    # Record inference start time
    start_time = time.time()
    # Perform model inference
    output = model(sensor_tensor, text_tensor)
    # Record inference end time
    end_time = time.time()
    # Calculate inference time (seconds)
    inference_time = end_time - start_time
    print(f"Model inference time: {inference_time:.6f} seconds")
    
    output = output.transpose(1, 3)  # Adjust output shape

# Denormalize and save the results
output = denormalize(output, mean, std)
output_np = output.cpu().numpy()
np.save('pred.npy', output_np)
print(f"Prediction results saved, shape: {output_np.shape}")