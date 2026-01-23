import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml


from mi_bminet_model.mi_bminet import MIBMINetDeploy

from testtraingenerate import load_config, load_train_config

def debug_specific_layer(target_layer_name, save_path="debug_output"):

    (pretrained, F1, D, F2, C, T, N, Nf, Nf2, activation, batch_size, opset_version) = load_config()
    lr = load_train_config()

    model = MIBMINetDeploy(
        F1=F1, D=D, F2=F2, C=C, T=T, N=N, Nf=Nf, Nf2=Nf2, activation=activation
    )
    model.train()


    storage = {
        "input": None,
        "output": None,
        "grad_input": None,
        "grad_output": None
    }

    def forward_hook(module, input, output):
   
        storage["input"] = input[0].detach().cpu().numpy()
        storage["output"] = output.detach().cpu().numpy()

    def backward_hook(module, grad_input, grad_output):
      
        storage["grad_output"] = grad_output[0].detach().cpu().numpy()
        if grad_input[0] is not None:
            storage["grad_input"] = grad_input[0].detach().cpu().numpy()

    try:
        layer = dict(model.named_modules())[target_layer_name]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)
        print(f"🎯 Successfully registered hooks on: {target_layer_name}")
    except KeyError:
        print(f"❌ Layer '{target_layer_name}' not found in model!")
        print("Available layers:", [name for name, _ in model.named_modules() if name])
        return

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, 1, C, T)
    labels = torch.randint(0, N, (batch_size,))
    criterion = nn.CrossEntropyLoss()


    pred = model(input_tensor)
    loss = criterion(pred, labels)
    loss.backward()


    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, f"debug_{target_layer_name}.npz")
    np.savez(
        file_name,
        x=storage["input"],           
        y=storage["output"],        
        gy=storage["grad_output"],   
        gx=storage["grad_input"]      
    )
    
    print(f"✅ Debug data saved to: {file_name}")
    print(f"   Shapes: x={storage['input'].shape}, y={storage['output'].shape}")

if __name__ == "__main__":

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "sep_conv1" 
    
    debug_specific_layer(target)