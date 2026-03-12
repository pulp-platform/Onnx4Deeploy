#!/bin/bash

python3 Onnx4Deeploy.py -operator PerturbNormal -o PerturbNormal
python3 Onnx4Deeploy.py -operator PerturbUniform -o PerturbUniform
python3 Onnx4Deeploy.py -operator PerturbRademacher -o PerturbRademacher
python3 Onnx4Deeploy.py -operator PerturbTriangle -o PerturbTriangle
python3 Onnx4Deeploy.py -operator PerturbEggroll -o PerturbEggroll
