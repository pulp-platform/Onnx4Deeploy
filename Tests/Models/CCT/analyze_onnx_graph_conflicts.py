#!/usr/bin/env python3
"""
ONNX Graph Connectivity Checker
检查ONNX图的连续性、完整性和潜在问题
"""

import onnx
import sys
from collections import defaultdict, deque
import argparse


class ONNXGraphChecker:
    def __init__(self, onnx_path):
        self.model = onnx.load(onnx_path)
        self.graph = self.model.graph
        self.onnx_path = onnx_path
        
        # 构建图的连接关系
        self.node_outputs = {}  # node_name -> [output_names]
        self.node_inputs = {}   # node_name -> [input_names]
        self.tensor_producers = {}  # tensor_name -> node_name
        self.tensor_consumers = defaultdict(list)  # tensor_name -> [node_names]
        self.all_tensors = set()
        
        self._build_graph_structure()
    
    def _build_graph_structure(self):
        """构建图的结构信息"""
        print(f"🔍 Analyzing ONNX graph from: {self.onnx_path}")
        print(f"📊 Graph has {len(self.graph.node)} nodes")
        
        # 记录所有初始化器
        initializers = {init.name for init in self.graph.initializer}
        print(f"📋 Graph has {len(initializers)} initializers")
        
        # 记录图输入输出
        graph_inputs = {inp.name for inp in self.graph.input}
        graph_outputs = {out.name for out in self.graph.output}
        print(f"🔌 Graph inputs: {len(graph_inputs)}, outputs: {len(graph_outputs)}")
        
        # 分析每个节点
        for i, node in enumerate(self.graph.node):
            node_name = node.name or f"node_{i}_{node.op_type}"
            
            # 记录节点的输入输出
            self.node_inputs[node_name] = list(node.input)
            self.node_outputs[node_name] = list(node.output)
            
            # 记录tensor的生产者和消费者
            for output_tensor in node.output:
                if output_tensor:  # 忽略空字符串
                    self.tensor_producers[output_tensor] = node_name
                    self.all_tensors.add(output_tensor)
            
            for input_tensor in node.input:
                if input_tensor:  # 忽略空字符串
                    self.tensor_consumers[input_tensor].append(node_name)
                    self.all_tensors.add(input_tensor)
        
        # 添加图级别的输入输出
        for inp in graph_inputs:
            self.all_tensors.add(inp)
        for out in graph_outputs:
            self.all_tensors.add(out)
    
    def check_connectivity(self):
        """检查图的连续性"""
        print(f"\n{'='*60}")
        print("🔗 CONNECTIVITY ANALYSIS")
        print(f"{'='*60}")
        
        missing_tensors = []
        orphaned_tensors = []
        broken_connections = []
        
        # 1. 检查缺失的tensor
        for tensor_name in self.all_tensors:
            has_producer = (tensor_name in self.tensor_producers or 
                          tensor_name in {inp.name for inp in self.graph.input} or
                          tensor_name in {init.name for init in self.graph.initializer})
            has_consumer = tensor_name in self.tensor_consumers or tensor_name in {out.name for out in self.graph.output}
            
            if not has_producer:
                missing_tensors.append(tensor_name)
            elif not has_consumer:
                orphaned_tensors.append(tensor_name)
        
        # 2. 检查断裂的连接
        for node_name, inputs in self.node_inputs.items():
            for input_tensor in inputs:
                if input_tensor and input_tensor not in self.tensor_producers and \
                   input_tensor not in {inp.name for inp in self.graph.input} and \
                   input_tensor not in {init.name for init in self.graph.initializer}:
                    broken_connections.append((node_name, input_tensor))
        
        # 报告结果
        if missing_tensors:
            print(f"❌ Found {len(missing_tensors)} missing tensors:")
            for tensor in missing_tensors:
                print(f"   - {tensor}")
                if tensor in self.tensor_consumers:
                    print(f"     Needed by: {self.tensor_consumers[tensor]}")
        else:
            print("✅ No missing tensors found")
        
        if orphaned_tensors:
            print(f"\n⚠️  Found {len(orphaned_tensors)} orphaned tensors:")
            for tensor in orphaned_tensors[:10]:  # 只显示前10个
                producer = self.tensor_producers.get(tensor, "unknown")
                print(f"   - {tensor} (produced by: {producer})")
            if len(orphaned_tensors) > 10:
                print(f"   ... and {len(orphaned_tensors) - 10} more")
        else:
            print("✅ No orphaned tensors found")
        
        if broken_connections:
            print(f"\n❌ Found {len(broken_connections)} broken connections:")
            for node, tensor in broken_connections:
                print(f"   - Node '{node}' expects tensor '{tensor}' but it's not produced")
        else:
            print("✅ No broken connections found")
        
        return len(missing_tensors) == 0 and len(broken_connections) == 0
    
    def find_similar_tensors(self, pattern):
        """查找匹配模式的tensor名称"""
        print(f"\n{'='*60}")
        print(f"🔍 SEARCHING FOR TENSORS MATCHING: {pattern}")
        print(f"{'='*60}")
        
        matches = []
        for tensor in self.all_tensors:
            if pattern.lower() in tensor.lower():
                matches.append(tensor)
        
        if matches:
            print(f"✅ Found {len(matches)} matching tensors:")
            for tensor in sorted(matches):
                print(f"   - {tensor}")
                if tensor in self.tensor_producers:
                    producer = self.tensor_producers[tensor]
                    print(f"     Producer: {producer}")
                if tensor in self.tensor_consumers:
                    consumers = self.tensor_consumers[tensor]
                    print(f"     Consumers: {consumers}")
                print()
        else:
            print(f"❌ No tensors found matching pattern: {pattern}")
        
        return matches
    
    def analyze_specific_tensor(self, tensor_name):
        """分析特定tensor的连接情况"""
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING TENSOR: {tensor_name}")
        print(f"{'='*60}")
        
        # 检查是否存在
        if tensor_name not in self.all_tensors:
            print(f"❌ Tensor '{tensor_name}' not found in graph!")
            
            # 寻找相似名称 - 更智能的匹配
            base_name = tensor_name.replace('__', '_').replace('_0', '').replace('_1', '')
            similar = []
            
            for t in self.all_tensors:
                # 完全匹配替换下划线
                if tensor_name.replace('__', '_') == t:
                    similar.append(t)
                # 或者基础名称匹配
                elif base_name in t and ('Add' in t or 'linear' in t):
                    similar.append(t)
            
            if similar:
                print(f"🔍 Similar tensor names found:")
                for sim in similar:
                    print(f"   - {sim}")
                    if sim in self.tensor_producers:
                        print(f"     Producer: {self.tensor_producers[sim]}")
            else:
                print(f"🔍 Searching for any tensor with 'linear1' and 'Add'...")
                self.find_similar_tensors("linear1")
                
            return False
        
        # 检查生产者
        if tensor_name in self.tensor_producers:
            producer = self.tensor_producers[tensor_name]
            print(f"✅ Producer: {producer}")
            
            # 显示生产者节点的详细信息
            if producer in self.node_outputs:
                print(f"   Producer outputs: {self.node_outputs[producer]}")
            if producer in self.node_inputs:
                print(f"   Producer inputs: {self.node_inputs[producer]}")
        elif tensor_name in {inp.name for inp in self.graph.input}:
            print(f"✅ This is a graph input")
        elif tensor_name in {init.name for init in self.graph.initializer}:
            print(f"✅ This is an initializer")
        else:
            print(f"❌ No producer found!")
        
        # 检查消费者
        if tensor_name in self.tensor_consumers:
            consumers = self.tensor_consumers[tensor_name]
            print(f"✅ Consumers ({len(consumers)}): {consumers}")
        elif tensor_name in {out.name for out in self.graph.output}:
            print(f"✅ This is a graph output")
        else:
            print(f"⚠️  No consumers found (orphaned tensor)")
        
        return True
    
    def find_path(self, start_tensor, end_tensor):
        """寻找两个tensor之间的路径"""
        print(f"\n{'='*60}")
        print(f"🛤️  FINDING PATH: {start_tensor} → {end_tensor}")
        print(f"{'='*60}")
        
        if start_tensor not in self.all_tensors:
            print(f"❌ Start tensor '{start_tensor}' not found!")
            return False
        
        if end_tensor not in self.all_tensors:
            print(f"❌ End tensor '{end_tensor}' not found!")
            return False
        
        # BFS搜索路径
        queue = deque([(start_tensor, [start_tensor])])
        visited = {start_tensor}
        
        while queue:
            current_tensor, path = queue.popleft()
            
            if current_tensor == end_tensor:
                print(f"✅ Found path with {len(path)} steps:")
                for i, tensor in enumerate(path):
                    if i < len(path) - 1:
                        # 找到生产下一个tensor的节点
                        next_tensor = path[i + 1]
                        producer = self.tensor_producers.get(next_tensor, "unknown")
                        print(f"   {i+1}. {tensor} → [{producer}] → {next_tensor}")
                    else:
                        print(f"   {i+1}. {tensor}")
                return True
            
            # 找到当前tensor的消费者节点，然后找这些节点的输出
            if current_tensor in self.tensor_consumers:
                for consumer_node in self.tensor_consumers[current_tensor]:
                    if consumer_node in self.node_outputs:
                        for output_tensor in self.node_outputs[consumer_node]:
                            if output_tensor and output_tensor not in visited:
                                visited.add(output_tensor)
                                queue.append((output_tensor, path + [output_tensor]))
        
        print(f"❌ No path found from {start_tensor} to {end_tensor}")
        return False
    
    def generate_report(self):
        """生成完整的图分析报告"""
        print(f"\n{'='*60}")
        print("📋 COMPREHENSIVE GRAPH REPORT")
        print(f"{'='*60}")
        
        # 基本统计
        print(f"📊 Basic Statistics:")
        print(f"   - Total nodes: {len(self.graph.node)}")
        print(f"   - Total tensors: {len(self.all_tensors)}")
        print(f"   - Initializers: {len(self.graph.initializer)}")
        print(f"   - Graph inputs: {len(self.graph.input)}")
        print(f"   - Graph outputs: {len(self.graph.output)}")
        
        # 节点类型统计
        op_types = defaultdict(int)
        for node in self.graph.node:
            op_types[node.op_type] += 1
        
        print(f"\n🔧 Node Types:")
        for op_type, count in sorted(op_types.items()):
            print(f"   - {op_type}: {count}")
        
        # 连续性检查
        is_connected = self.check_connectivity()
        
        return is_connected


def main():
    parser = argparse.ArgumentParser(description="Check ONNX graph connectivity")
    parser.add_argument("onnx_file", help="Path to ONNX file")
    parser.add_argument("--tensor", help="Analyze specific tensor")
    parser.add_argument("--search", help="Search for tensors matching pattern")
    parser.add_argument("--path", nargs=2, help="Find path between two tensors", metavar=("START", "END"))
    parser.add_argument("--report", action="store_true", help="Generate full report")
    
    args = parser.parse_args()
    
    try:
        checker = ONNXGraphChecker(args.onnx_file)
        
        if args.tensor:
            checker.analyze_specific_tensor(args.tensor)
        elif args.search:
            checker.find_similar_tensors(args.search)
        elif args.path:
            start, end = args.path
            checker.find_path(start, end)
        elif args.report:
            checker.generate_report()
        else:
            # 默认只检查连续性
            is_connected = checker.check_connectivity()
            if is_connected:
                print(f"\n✅ Graph is properly connected!")
            else:
                print(f"\n❌ Graph has connectivity issues!")
            
    except Exception as e:
        print(f"❌ Error analyzing graph: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())