# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from glob import glob

import torch
from tqdm import tqdm

import safetensor_convert as sc

def convert(src_path):
    #os.makedirs(dst_path, exist_ok=True)
    
    for file_path in tqdm(glob(os.path.join(src_path, "*.*"))):
        dir_path, file_name_with_ext = os.path.split(file_path)
        file_name = os.path.basename(file_path)
        
        if file_path.endswith('.safetensors'):
            dst_file = file_path.replace(".safetensors", ".bin")
        
            sc.binary_output_file =dst_file
            sc.safetensor_folder = dir_path
            sc.save_safetensor_weights_to_binary(file_path,sc.binary_output_file)
       
        
        # 强制释放内存
        if 'weights' in locals():
            del weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert every safetensors file in a directory to Garnet binary format.")
    parser.add_argument("source", help="Directory containing safetensors files")
    args = parser.parse_args()
    convert(args.source)
