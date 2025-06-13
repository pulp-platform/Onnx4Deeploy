#!/usr/bin/env python3
"""
ONNX模型详细分析报告生成器
生成包含所有参数和激活值的详细报告
"""

import onnx
import numpy as np
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import os


class ONNXDetailedReporter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.report_data = {
            'model_info': {},
            'parameters': [],
            'activations': [],
            'nodes': [],
            'summary': {}
        }
        
    def get_dtype_name(self, dtype_int: int) -> str:
        """获取数据类型名称"""
        dtype_map = {
            1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16', 5: 'int16',
            6: 'int32', 7: 'int64', 8: 'string', 9: 'bool', 10: 'float16',
            11: 'float64', 12: 'uint32', 13: 'uint64', 14: 'complex64', 15: 'complex128'
        }
        return dtype_map.get(dtype_int, f'unknown_{dtype_int}')
    
    def calculate_size(self, shape: List[int], dtype: str) -> int:
        """计算tensor字节大小"""
        dtype_sizes = {
            'float32': 4, 'float': 4, 'float64': 8, 'double': 8, 'float16': 2, 'half': 2,
            'int32': 4, 'int': 4, 'int64': 8, 'long': 8, 'int16': 2, 'short': 2,
            'int8': 1, 'char': 1, 'uint8': 1, 'uchar': 1, 'uint16': 2, 'ushort': 2,
            'uint32': 4, 'uint': 4, 'uint64': 8, 'ulong': 8, 'bool': 1, 'string': 1
        }
        
        if not shape or any(dim <= 0 for dim in shape):
            return 0
        
        element_count = np.prod([dim if dim > 0 else 1 for dim in shape])
        element_size = dtype_sizes.get(dtype.lower(), 4)
        return int(element_count * element_size)
    
    def format_bytes(self, bytes_count: int) -> str:
        """格式化字节数"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.2f} TB"
    
    def analyze_model_info(self):
        """分析模型基本信息"""
        print("📋 分析模型基本信息...")
        
        model_info = {
            'model_path': self.model_path,
            'model_version': getattr(self.model, 'model_version', 'unknown'),
            'producer_name': getattr(self.model, 'producer_name', 'unknown'),
            'producer_version': getattr(self.model, 'producer_version', 'unknown'),
            'domain': getattr(self.model, 'domain', 'unknown'),
            'ir_version': getattr(self.model, 'ir_version', 'unknown'),
            'doc_string': getattr(self.model, 'doc_string', ''),
            'file_size': os.path.getsize(self.model_path),
            'file_size_formatted': self.format_bytes(os.path.getsize(self.model_path)),
            'analysis_time': datetime.now().isoformat()
        }
        
        # 统计节点类型
        node_types = {}
        for node in self.graph.node:
            node_type = node.op_type
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        model_info['node_statistics'] = node_types
        model_info['total_nodes'] = len(self.graph.node)
        model_info['total_inputs'] = len(self.graph.input)
        model_info['total_outputs'] = len(self.graph.output)
        model_info['total_initializers'] = len(self.graph.initializer)
        model_info['total_value_infos'] = len(self.graph.value_info)
        
        self.report_data['model_info'] = model_info
    
    def analyze_parameters(self):
        """分析所有参数"""
        print("⚖️ 分析模型参数...")
        
        parameters = []
        total_param_count = 0
        total_param_size = 0
        
        for i, initializer in enumerate(self.graph.initializer):
            name = initializer.name
            shape = list(initializer.dims)
            dtype = self.get_dtype_name(initializer.data_type)
            
            param_count = np.prod(shape) if shape else 0
            param_size = self.calculate_size(shape, dtype)
            
            # 获取实际数据（如果需要）
            try:
                data_array = onnx.numpy_helper.to_array(initializer)
                actual_shape = list(data_array.shape)
                actual_dtype = str(data_array.dtype)
                min_val = float(data_array.min()) if data_array.size > 0 else 0
                max_val = float(data_array.max()) if data_array.size > 0 else 0
                mean_val = float(data_array.mean()) if data_array.size > 0 else 0
                std_val = float(data_array.std()) if data_array.size > 0 else 0
            except:
                actual_shape = shape
                actual_dtype = dtype
                min_val = max_val = mean_val = std_val = None
            
            param_info = {
                'index': i,
                'name': name,
                'shape': shape,
                'actual_shape': actual_shape,
                'dtype': dtype,
                'actual_dtype': actual_dtype,
                'param_count': int(param_count),
                'size_bytes': param_size,
                'size_formatted': self.format_bytes(param_size),
                'min_value': min_val,
                'max_value': max_val,
                'mean_value': mean_val,
                'std_value': std_val
            }
            
            parameters.append(param_info)
            total_param_count += param_count
            total_param_size += param_size
        
        self.report_data['parameters'] = parameters
        self.report_data['summary']['total_param_count'] = int(total_param_count)
        self.report_data['summary']['total_param_size'] = total_param_size
        self.report_data['summary']['total_param_size_formatted'] = self.format_bytes(total_param_size)
    
    def analyze_activations(self):
        """分析激活值"""
        print("🔄 分析激活值...")
        
        activations = []
        total_activation_size = 0
        
        # 收集所有tensor信息
        param_names = {init.name for init in self.graph.initializer}
        
        # 从value_info获取形状信息
        value_info_dict = {vi.name: vi for vi in self.graph.value_info}
        input_dict = {inp.name: inp for inp in self.graph.input}
        output_dict = {out.name: out for out in self.graph.output}
        
        # 分析每个节点的输出
        activation_index = 0
        for node_idx, node in enumerate(self.graph.node):
            for output_idx, output_name in enumerate(node.output):
                if output_name not in param_names:  # 排除参数
                    shape = None
                    dtype = 'float32'
                    
                    # 尝试从各个来源获取形状信息
                    tensor_info = None
                    source = 'unknown'
                    
                    if output_name in value_info_dict:
                        tensor_info = value_info_dict[output_name]
                        source = 'value_info'
                    elif output_name in input_dict:
                        tensor_info = input_dict[output_name]
                        source = 'input'
                    elif output_name in output_dict:
                        tensor_info = output_dict[output_name]
                        source = 'output'
                    
                    if tensor_info and tensor_info.type.tensor_type.shape:
                        try:
                            shape = []
                            for dim in tensor_info.type.tensor_type.shape.dim:
                                if dim.dim_value > 0:
                                    shape.append(dim.dim_value)
                                elif dim.dim_param:
                                    shape.append(-1)  # 动态维度
                                else:
                                    shape.append(1)
                            dtype = self.get_dtype_name(tensor_info.type.tensor_type.elem_type)
                        except:
                            pass
                    
                    # 计算大小
                    if shape:
                        actual_shape = [1 if dim == -1 else dim for dim in shape]
                        activation_size = self.calculate_size(actual_shape, dtype)
                    else:
                        activation_size = 0
                    
                    activation_info = {
                        'index': activation_index,
                        'name': output_name,
                        'node_index': node_idx,
                        'node_name': node.name if node.name else f"node_{node_idx}",
                        'node_type': node.op_type,
                        'output_index': output_idx,
                        'shape': shape,
                        'dtype': dtype,
                        'size_bytes': activation_size,
                        'size_formatted': self.format_bytes(activation_size),
                        'info_source': source,
                        'is_dynamic': shape and (-1 in shape) if shape else False
                    }
                    
                    activations.append(activation_info)
                    total_activation_size += activation_size
                    activation_index += 1
        
        self.report_data['activations'] = activations
        self.report_data['summary']['total_activation_count'] = len(activations)
        self.report_data['summary']['total_activation_size'] = total_activation_size
        self.report_data['summary']['total_activation_size_formatted'] = self.format_bytes(total_activation_size)
    
    def analyze_nodes(self):
        """分析所有节点"""
        print("🔗 分析计算节点...")
        
        nodes = []
        for i, node in enumerate(self.graph.node):
            node_info = {
                'index': i,
                'name': node.name if node.name else f"node_{i}",
                'op_type': node.op_type,
                'domain': node.domain,
                'input_count': len(node.input),
                'output_count': len(node.output),
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {}
            }
            
            # 解析属性
            for attr in node.attribute:
                attr_name = attr.name
                attr_type = attr.type
                
                try:
                    if attr_type == onnx.AttributeProto.INT:
                        node_info['attributes'][attr_name] = attr.i
                    elif attr_type == onnx.AttributeProto.FLOAT:
                        node_info['attributes'][attr_name] = attr.f
                    elif attr_type == onnx.AttributeProto.STRING:
                        node_info['attributes'][attr_name] = attr.s.decode('utf-8')
                    elif attr_type == onnx.AttributeProto.INTS:
                        node_info['attributes'][attr_name] = list(attr.ints)
                    elif attr_type == onnx.AttributeProto.FLOATS:
                        node_info['attributes'][attr_name] = list(attr.floats)
                    elif attr_type == onnx.AttributeProto.STRINGS:
                        node_info['attributes'][attr_name] = [s.decode('utf-8') for s in attr.strings]
                    else:
                        node_info['attributes'][attr_name] = f"<{attr_type}>"
                except:
                    node_info['attributes'][attr_name] = "<parse_error>"
            
            nodes.append(node_info)
        
        self.report_data['nodes'] = nodes
    
    def generate_summary(self):
        """生成汇总信息"""
        print("📊 生成汇总信息...")
        
        summary = self.report_data['summary']
        
        # 计算总内存占用
        total_memory = summary.get('total_param_size', 0) + summary.get('total_activation_size', 0)
        summary['total_memory_estimate'] = total_memory
        summary['total_memory_estimate_formatted'] = self.format_bytes(total_memory)
        
        # 参数统计
        if self.report_data['parameters']:
            param_sizes = [p['size_bytes'] for p in self.report_data['parameters']]
            summary['largest_param_size'] = max(param_sizes)
            summary['largest_param_size_formatted'] = self.format_bytes(max(param_sizes))
            summary['smallest_param_size'] = min(param_sizes)
            summary['smallest_param_size_formatted'] = self.format_bytes(min(param_sizes))
            summary['avg_param_size'] = sum(param_sizes) / len(param_sizes)
            summary['avg_param_size_formatted'] = self.format_bytes(int(summary['avg_param_size']))
        
        # 激活值统计
        if self.report_data['activations']:
            activation_sizes = [a['size_bytes'] for a in self.report_data['activations'] if a['size_bytes'] > 0]
            if activation_sizes:
                summary['largest_activation_size'] = max(activation_sizes)
                summary['largest_activation_size_formatted'] = self.format_bytes(max(activation_sizes))
                summary['smallest_activation_size'] = min(activation_sizes)
                summary['smallest_activation_size_formatted'] = self.format_bytes(min(activation_sizes))
                summary['avg_activation_size'] = sum(activation_sizes) / len(activation_sizes)
                summary['avg_activation_size_formatted'] = self.format_bytes(int(summary['avg_activation_size']))
    
    def save_json_report(self, output_path: str):
        """保存JSON格式报告"""
        print(f"💾 保存JSON报告到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
    
    def save_csv_reports(self, base_path: str):
        """保存CSV格式报告"""
        print(f"💾 保存CSV报告到: {base_path}_*.csv")
        
        # 参数CSV
        params_csv = f"{base_path}_parameters.csv"
        with open(params_csv, 'w', newline='', encoding='utf-8') as f:
            if self.report_data['parameters']:
                writer = csv.DictWriter(f, fieldnames=self.report_data['parameters'][0].keys())
                writer.writeheader()
                writer.writerows(self.report_data['parameters'])
        
        # 激活值CSV
        activations_csv = f"{base_path}_activations.csv"
        with open(activations_csv, 'w', newline='', encoding='utf-8') as f:
            if self.report_data['activations']:
                writer = csv.DictWriter(f, fieldnames=self.report_data['activations'][0].keys())
                writer.writeheader()
                writer.writerows(self.report_data['activations'])
        
        # 节点CSV
        nodes_csv = f"{base_path}_nodes.csv"
        with open(nodes_csv, 'w', newline='', encoding='utf-8') as f:
            if self.report_data['nodes']:
                # 简化节点信息用于CSV
                simplified_nodes = []
                for node in self.report_data['nodes']:
                    simplified = {k: v for k, v in node.items() if k != 'attributes'}
                    simplified['attributes_count'] = len(node.get('attributes', {}))
                    simplified_nodes.append(simplified)
                
                writer = csv.DictWriter(f, fieldnames=simplified_nodes[0].keys())
                writer.writeheader()
                writer.writerows(simplified_nodes)
    
    def save_text_report(self, output_path: str):
        """保存文本格式详细报告"""
        print(f"💾 保存文本报告到: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ONNX模型详细分析报告\n")
            f.write("="*80 + "\n\n")
            
            # 模型基本信息
            f.write("📋 模型基本信息\n")
            f.write("-"*40 + "\n")
            model_info = self.report_data['model_info']
            for key, value in model_info.items():
                if key != 'node_statistics':
                    f.write(f"{key}: {value}\n")
            
            f.write("\n节点类型统计:\n")
            for node_type, count in model_info.get('node_statistics', {}).items():
                f.write(f"  {node_type}: {count}\n")
            
            # 汇总信息
            f.write(f"\n📊 汇总信息\n")
            f.write("-"*40 + "\n")
            summary = self.report_data['summary']
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            # 详细参数列表
            f.write(f"\n⚖️ 详细参数列表 (共{len(self.report_data['parameters'])}个)\n")
            f.write("-"*40 + "\n")
            for param in self.report_data['parameters']:
                f.write(f"\n参数 #{param['index']}: {param['name']}\n")
                f.write(f"  形状: {param['shape']}\n")
                f.write(f"  数据类型: {param['dtype']}\n")
                f.write(f"  参数数量: {param['param_count']:,}\n")
                f.write(f"  内存大小: {param['size_formatted']}\n")
                if param['min_value'] is not None:
                    f.write(f"  数值范围: [{param['min_value']:.6f}, {param['max_value']:.6f}]\n")
                    f.write(f"  均值: {param['mean_value']:.6f}, 标准差: {param['std_value']:.6f}\n")
            
            # 详细激活值列表
            f.write(f"\n🔄 详细激活值列表 (共{len(self.report_data['activations'])}个)\n")
            f.write("-"*40 + "\n")
            for activation in self.report_data['activations']:
                f.write(f"\n激活值 #{activation['index']}: {activation['name']}\n")
                f.write(f"  来源节点: {activation['node_name']} ({activation['node_type']})\n")
                f.write(f"  形状: {activation['shape']}\n")
                f.write(f"  数据类型: {activation['dtype']}\n")
                f.write(f"  内存大小: {activation['size_formatted']}\n")
                f.write(f"  信息来源: {activation['info_source']}\n")
                if activation['is_dynamic']:
                    f.write(f"  包含动态维度: 是\n")
            
            # 详细节点列表
            f.write(f"\n🔗 详细节点列表 (共{len(self.report_data['nodes'])}个)\n")
            f.write("-"*40 + "\n")
            for node in self.report_data['nodes']:
                f.write(f"\n节点 #{node['index']}: {node['name']}\n")
                f.write(f"  操作类型: {node['op_type']}\n")
                f.write(f"  输入: {node['inputs']}\n")
                f.write(f"  输出: {node['outputs']}\n")
                if node['attributes']:
                    f.write(f"  属性:\n")
                    for attr_name, attr_value in node['attributes'].items():
                        f.write(f"    {attr_name}: {attr_value}\n")
    
    def generate_full_report(self, output_prefix: str):
        """生成完整报告"""
        print(f"🚀 开始生成详细报告...")
        print(f"模型文件: {self.model_path}")
        print("="*60)
        
        # 执行所有分析
        self.analyze_model_info()
        self.analyze_parameters()
        self.analyze_activations()
        self.analyze_nodes()
        self.generate_summary()
        
        # 保存各种格式的报告
        self.save_json_report(f"{output_prefix}.json")
        self.save_csv_reports(output_prefix)
        self.save_text_report(f"{output_prefix}.txt")
        
        print("\n✅ 报告生成完成!")
        print(f"生成的文件:")
        print(f"  - {output_prefix}.json (JSON格式完整报告)")
        print(f"  - {output_prefix}.txt (文本格式详细报告)")
        print(f"  - {output_prefix}_parameters.csv (参数CSV)")
        print(f"  - {output_prefix}_activations.csv (激活值CSV)")
        print(f"  - {output_prefix}_nodes.csv (节点CSV)")


def main():
    parser = argparse.ArgumentParser(description='生成ONNX模型详细分析报告')
    parser.add_argument('model_path', help='ONNX模型文件路径')
    parser.add_argument('-o', '--output', default='onnx_report', 
                       help='输出文件前缀 (默认: onnx_report)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误: 文件 {args.model_path} 不存在")
        return
    
    try:
        reporter = ONNXDetailedReporter(args.model_path)
        reporter.generate_full_report(args.output)
    except Exception as e:
        print(f"生成报告时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()