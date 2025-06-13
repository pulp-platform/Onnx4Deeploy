#!/usr/bin/env python3
"""
CCT Training Mode Comparison Script
This script helps compare the three different training modes:
- Linear Probe: Only classifier
- Last Attention: Classifier + last attention block  
- Both Attention: Classifier + all attention blocks
"""

import numpy as np
import os
from pathlib import Path

def load_and_compare_data():
    """Load and compare data from all three training modes"""
    
    modes = ['linear_probe', 'last_attention', 'both_attention']
    base_dir = Path(__file__).parent
    
    print("🔍 Comparing training data across modes...")
    
    for mode in modes:
        mode_dir = base_dir / f"onnx/CCT_{mode}_32_128_4_2"  # Adjust based on your config
        
        if not mode_dir.exists():
            print(f"❌ Directory not found: {mode_dir}")
            continue
            
        print(f"\n📊 {mode.upper()} MODE:")
        
        # Check input data (following original naming)
        input_file = mode_dir / "inputs.npz"
        if input_file.exists():
            data = np.load(input_file)
            for key in data.files:
                input_shape = data[key].shape
                print(f"   Input '{key}' shape: {input_shape}")
                print(f"   Input '{key}' range: [{data[key].min():.3f}, {data[key].max():.3f}]")
        
        # Check output data  
        output_file = mode_dir / "outputs.npz"
        if output_file.exists():
            data = np.load(output_file)
            for key in data.files:
                output_shape = data[key].shape
                print(f"   Output '{key}' shape: {output_shape}")
        
        # Check ONNX files
        onnx_files = ['network_infer.onnx', 'network_train.onnx', 'network.onnx']
        for onnx_file in onnx_files:
            file_path = mode_dir / onnx_file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   ONNX '{onnx_file}': {size_mb:.2f} MB")

if __name__ == "__main__":
    load_and_compare_data()
