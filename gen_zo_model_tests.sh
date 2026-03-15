#!/bin/bash

# LiteCNN
python3 Onnx4Deeploy.py -model LightweightCNN -mode infer -o LiteCNN
python3 Onnx4Deeploy.py -model LightweightCNN -mode zo-train -o LiteCNN-Rad --noise-type rademacher
python3 Onnx4Deeploy.py -model LightweightCNN -mode zo-train -o LiteCNN-Lorp --noise-type eggroll
python3 Onnx4Deeploy.py -model LightweightCNN -mode zo-train -o LiteCNN-Uniform --noise-type uniform
python3 Onnx4Deeploy.py -model LightweightCNN -mode zo-train -o LiteCNN-Gaussian --noise-type gaussian

# QLiteCNN
python3 Onnx4Deeploy.py -model QLiteCNN -mode q-infer -o QLiteCNN
python3 Onnx4Deeploy.py -model QLiteCNN -mode q-zo-train -o QLiteCNN-Rad --noise-type rademacher
python3 Onnx4Deeploy.py -model QLiteCNN -mode q-zo-train -o QLiteCNN-Lorp --noise-type eggroll
python3 Onnx4Deeploy.py -model QLiteCNN -mode q-zo-train -o QLiteCNN-Uniform --noise-type uniform
python3 Onnx4Deeploy.py -model QLiteCNN -mode q-zo-train -o QLiteCNN-Gaussian --noise-type gaussian