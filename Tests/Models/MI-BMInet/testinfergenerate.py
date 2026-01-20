import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import yaml
from mi_bminet_model.mi_bminet import *
from utils.utils import *

# ==========================================
# 1. 参数统计辅助工具
# ==========================================
def print_model_summary(model, input_size):
    """
    打印模型每一层的详细信息、参数量和特征图维度
    """
    print("\n" + "="*80)
    print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
    print("-"*80)
    
    total_params = 0
    trainable_params = 0
    
    # 钩子函数用于获取每一层的输出形状
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {}
            summary[m_key]["output_shape"] = list(output.shape)
            
            params = 0
            if hasattr(module, "weight") and module.weight is not None:
                params += torch.numel(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                params += torch.numel(module.bias)
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    model.apply(register_hook)

    # 构造一个虚拟输入来触发 hook
    x = torch.zeros(input_size).to(next(model.parameters()).device)
    try:
        model(x)
    except Exception as e:
        print(f"统计输出维度时出错 (可能是Flatten层): {e}")

    for layer in summary:
        print(f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['nb_params']:<15,}")
        total_params += summary[layer]["nb_params"]

    # 移除 hooks
    for h in hooks:
        h.remove()

    print("-"*80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_params:,}")
    print(f"Estimated Total Size (FP32): {total_params * 4 / 1024:.2f} KB")
    print("="*80 + "\n")

# ==========================================
# 2. 配置加载
# ==========================================
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mi = config["mi_bminet"]
    return (
        mi["pretrained"], mi["F1"], mi["D"], mi.get("F2", mi["F1"]*mi["D"]),
        mi["C"], mi["T"], mi["N"], mi["Nf"], mi["Nf2"],
        mi["p_dropout"], mi["activation"], mi["opset_version"]
    )

# ==========================================
# 3. 主导出与打印函数
# ==========================================
def generate_mi_bminet_onnx_and_data(save_path=None):
    # 加载配置
    pretrained, F1, D, F2, C, T, N, Nf, Nf2, p_dropout, activation, opset_version = load_config()
    input_shape = (1, 1, C, T)

    print(f"✅ Loaded Config: EEG_Channels={C}, Time_Samples={T}, Classes={N}")

    # 实例化部署模型
    from mi_bminet_model.mi_bminet import MIBMINetDeploy, MIBMINet
    
    # 默认创建部署模型
    model = MIBMINetDeploy(
        F1=F1, D=D, F2=F2, C=C, T=T, N=N, 
        Nf=Nf, Nf2=Nf2, activation=activation
    )

    # 如果有预训练权重则加载并融合BN
    if pretrained:
        print(f"📦 Loading pretrained weights from {pretrained}...")
        try:
            training_model = MIBMINet(F1=F1, D=D, F2=F2, C=C, T=T, N=N, Nf=Nf, Nf2=Nf2)
            # 假设你的权重加载逻辑
            training_model.load_state_dict(torch.load(pretrained, map_location='cpu'))
            model.load_and_fuse_from_training_model(training_model)
            print("✅ BatchNorm fused into Conv layers.")
        except:
            print("⚠️ Pretrained load failed, using random weights.")

    model.eval()

    # --- 打印模型每一层的信息 ---
    print_model_summary(model, input_shape)

    # 路径准备
    folder_name = f"MI_BMInet_C{C}_T{T}_N{N}"
    base_path = save_path if save_path else os.path.join(os.path.dirname(__file__), "onnx", folder_name)
    os.makedirs(base_path, exist_ok=True)
    onnx_file = os.path.join(base_path, "network.onnx")

    # 导出 ONNX
    print(f"🔧 Exporting to ONNX (Opset {opset_version})...")
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model, dummy_input, onnx_file,
        opset_version=opset_version,
        input_names=["input"], output_names=["output"]
    )

    # 优化与简化
    import onnxsim
    model_sim, check = onnxsim.simplify(onnx_file)
    if check:
        onnx.save(model_sim, onnx_file)
        print(f"✅ ONNX Simplified & Saved: {onnx_file}")

    # 保存测试输入输出
    input_data = dummy_input.numpy().astype(np.float32)
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]
    
    np.savez(os.path.join(base_path, "inputs.npz"), input=input_data)
    np.savez(os.path.join(base_path, "outputs.npz"), output=output_data)
    print(f"✅ Verification data saved to {base_path}")

if __name__ == "__main__":
    generate_mi_bminet_onnx_and_data()