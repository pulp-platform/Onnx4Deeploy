#!/usr/bin/env python3
"""
诊断 GroupNorm 前向/反向节点的输入映射问题
"""
import onnx
import sys
import os

def diagnose_gn_mapping(onnx_path):
    """检查 GroupNorm 前向和反向节点的输入映射"""
    model = onnx.load(onnx_path)
    graph = model.graph

    print(f"\n{'='*60}")
    print(f"诊断文件: {onnx_path}")
    print(f"{'='*60}")

    # 收集前向 GroupNorm 节点的输入
    forward_gn_inputs = {}  # node_name -> X (input tensor name)
    forward_gn_outputs = {}  # Y (output tensor name) -> X (input tensor name)

    for node in graph.node:
        if node.op_type == "GroupNormalization":
            X = node.input[0]
            Y = node.output[0]
            forward_gn_inputs[node.name] = X
            forward_gn_outputs[Y] = X
            print(f"\n🔵 前向 GroupNorm: {node.name}")
            print(f"   输入 X: {X}")
            print(f"   输出 Y: {Y}")

    # 收集反向 GroupNormGrad 节点的输入
    print(f"\n{'-'*60}")

    for node in graph.node:
        if node.op_type == "GroupNormalizationGrad":
            dY = node.input[0]
            X = node.input[1]
            gamma = node.input[2]

            print(f"\n🔴 反向 GroupNormGrad: {node.name}")
            print(f"   输入 dY: {dY}")
            print(f"   输入 X: {X}")
            print(f"   输入 gamma: {gamma}")

            # 检查 X 是否在前向节点的输入中
            if X in forward_gn_inputs.values():
                matching_fwd = [k for k, v in forward_gn_inputs.items() if v == X]
                print(f"   ✅ X 与前向节点匹配: {matching_fwd}")
            else:
                print(f"   ❌ X 不在前向节点的输入中!")

                # 检查 X 是否是某个前向节点的输出
                if X in forward_gn_outputs:
                    original_X = forward_gn_outputs[X]
                    print(f"   💡 X ({X}) 是前向 GroupNorm 的输出!")
                    print(f"   💡 对应的前向输入应该是: {original_X}")
                    print(f"   💡 正确的 stat 张量应该是: {original_X}_stat_combined")
                else:
                    print(f"   ⚠️  无法找到对应的前向节点")

            # 显示输出
            outputs = [o for o in node.output if o]
            print(f"   输出: {outputs}")

    # 检查已拆分的节点
    print(f"\n{'-'*60}")
    print("检查已拆分的 GroupNormGradW 节点:")

    for node in graph.node:
        if node.op_type == "GroupNormGradW":
            dY = node.input[0]
            X = node.input[1]
            stat = node.input[2] if len(node.input) > 2 else "N/A"

            print(f"\n🟡 GroupNormGradW: {node.name}")
            print(f"   输入 dY: {dY}")
            print(f"   输入 X: {X}")
            print(f"   输入 stat: {stat}")

            # 检查 stat 张量是否存在
            stat_exists = any(vi.name == stat for vi in graph.value_info) or \
                          any(init.name == stat for init in graph.initializer)

            # 检查是否有对应的 GroupNormStats 节点生成这个 stat
            stat_producer = None
            for n in graph.node:
                if stat in n.output:
                    stat_producer = n.name
                    break

            if stat_producer:
                print(f"   ✅ stat 由节点 {stat_producer} 生成")
            else:
                print(f"   ❌ 找不到生成 stat ({stat}) 的节点!")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # 查找最新生成的 network.onnx
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_dir = os.path.join(base_dir, "onnx")

    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    else:
        # 查找 onnx 目录下的子目录
        if os.path.exists(onnx_dir):
            subdirs = [d for d in os.listdir(onnx_dir) if os.path.isdir(os.path.join(onnx_dir, d))]
            if subdirs:
                latest_dir = os.path.join(onnx_dir, subdirs[0])
                onnx_path = os.path.join(latest_dir, "network.onnx")
            else:
                print("未找到 onnx 子目录")
                sys.exit(1)
        else:
            print(f"目录不存在: {onnx_dir}")
            sys.exit(1)

    if os.path.exists(onnx_path):
        diagnose_gn_mapping(onnx_path)
    else:
        print(f"文件不存在: {onnx_path}")
