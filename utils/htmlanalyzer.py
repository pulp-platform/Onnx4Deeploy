#!/usr/bin/env python3
"""
Enhanced HTML Memory Allocation Analyzer

This script analyzes HTML files containing JavaScript memory allocation data,
identifies constant parameters, calculates activation+gradient memory, and generates
a comprehensive memory breakdown CSV with peak timing and gradient analysis.

Author: Assistant
Date: 2025
"""

import re
import json
import sys
import os
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


@dataclass
class MemoryComponent:
    """Data class to represent a memory component"""
    name: str
    memory_size: int
    y_start: int
    y_end: int
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    x_coords: Optional[List[float]] = None
    y_coords: Optional[List[float]] = None
    is_constant: bool = False
    waste_type: str = "normal"  # normal, duplicate, bias, broadcast


@dataclass
class MemoryBreakdown:
    """Data class for memory breakdown analysis"""
    total_memory: int
    constant_memory: int
    activation_gradient_memory: int
    peak_memory_usage: int
    strategy_name: str = ""
    peak_time: Optional[float] = None
    peak_components: Optional[List[str]] = None
    peak_gradient_memory: int = 0
    peak_gradient_count: int = 0
    peak_gradient_components: Optional[List[str]] = None


class HTMLMemoryAnalyzer:
    """Main analyzer class for processing HTML memory allocation data"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.components: List[MemoryComponent] = []
        self.global_x_min: Optional[float] = None
        self.global_x_max: Optional[float] = None
        self.memory_timeline: List[float] = []  # Track memory usage over time
        self.timeline_points: List[float] = []  # Time points corresponding to memory_timeline
    
    def load_html_file(self, file_path: str) -> str:
        """Load HTML file content"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found")
            
            print(f"Loading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if self.debug:
                print(f"File size: {len(content):,} characters")
            
            return content
        
        except Exception as e:
            print(f"Error loading file: {e}")
            return ""
    
    def parse_html_content(self, html_content: str) -> None:
        """Parse HTML content and extract memory components"""
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tags = soup.find_all('script')
        
        print(f"Found {len(script_tags)} script tags")
        
        for i, script in enumerate(script_tags):
            if script.string and ('data' in script.string or 'fig' in script.string):
                components = self._extract_components_from_script(script.string, i + 1)
                self.components.extend(components)
        
        print(f"Total components extracted: {len(self.components)}")
        
        # Calculate memory timeline for peak analysis
        self._calculate_memory_timeline()
    
    def _calculate_memory_timeline(self) -> None:
        """Calculate memory usage over time to find peak usage"""
        if not self.components:
            return
        
        # Get all unique x coordinates to create timeline
        all_x_coords = []
        for comp in self.components:
            if comp.x_coords:
                all_x_coords.extend(comp.x_coords)
        
        if not all_x_coords:
            return
        
        # Create timeline with regular intervals
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        self.timeline_points = np.linspace(x_min, x_max, 1000).tolist()
        
        # Calculate memory usage at each timeline point
        self.memory_timeline = []
        for x_point in self.timeline_points:
            total_memory_at_point = 0
            
            for comp in self.components:
                if comp.x_start is not None and comp.x_end is not None:
                    # Check if component is active at this time point
                    if comp.x_start <= x_point <= comp.x_end:
                        total_memory_at_point += comp.memory_size
            
            self.memory_timeline.append(total_memory_at_point)
    
    def get_peak_memory_usage(self) -> int:
        """Get peak memory usage from timeline"""
        return max(self.memory_timeline) if self.memory_timeline else 0
    
    def get_peak_memory_info(self) -> Tuple[int, Optional[float], List[str]]:
        """Get peak memory usage, time when it occurs, and active components"""
        if not self.memory_timeline:
            return 0, None, []
        
        # Find peak memory and its index
        peak_memory = max(self.memory_timeline)
        peak_index = self.memory_timeline.index(peak_memory)
        peak_time = self.timeline_points[peak_index] if peak_index < len(self.timeline_points) else None
        
        # Find components active at peak time
        active_components = []
        if peak_time is not None:
            for comp in self.components:
                if comp.x_start is not None and comp.x_end is not None:
                    if comp.x_start <= peak_time <= comp.x_end:
                        active_components.append(comp.name)
        
        return peak_memory, peak_time, active_components
    
    def get_peak_gradient_memory_info(self) -> Tuple[int, int, List[str]]:
        """Get gradient memory info at peak time"""
        if not self.memory_timeline:
            return 0, 0, []
        
        # Find peak memory and its index
        peak_memory = max(self.memory_timeline)
        peak_index = self.memory_timeline.index(peak_memory)
        peak_time = self.timeline_points[peak_index] if peak_index < len(self.timeline_points) else None
        
        if peak_time is None:
            return 0, 0, []
        
        # Find gradient components active at peak time
        gradient_components = []
        total_gradient_memory = 0
        gradient_component_count = 0
        
        for comp in self.components:
            if comp.x_start is not None and comp.x_end is not None:
                if comp.x_start <= peak_time <= comp.x_end:
                    # Check if component name contains gradient-related keywords
                    comp_name_lower = comp.name.lower()
                    if ('grad' in comp_name_lower or 'updated' in comp_name_lower):
                        gradient_components.append(comp.name)
                        total_gradient_memory += comp.memory_size
                        gradient_component_count += 1
        
        return total_gradient_memory, gradient_component_count, gradient_components
    
    def _extract_components_from_script(self, script_content: str, script_index: int) -> List[MemoryComponent]:
        """Extract memory components from a single script tag"""
        components = []
        
        if self.debug:
            print(f"Processing script {script_index}, length: {len(script_content)}")
        
        # Try JSON parsing first
        components = self._try_json_parsing(script_content)
        
        # Fallback to regex if JSON parsing fails
        if not components:
            components = self._try_regex_parsing(script_content)
        
        if components:
            print(f"Extracted {len(components)} components from script {script_index}")
        
        return components
    
    def _try_json_parsing(self, script_content: str) -> List[MemoryComponent]:
        """Try to parse JavaScript fig object as JSON"""
        fig_match = re.search(r'var\s+fig\s*=\s*(\{.*?\});', script_content, re.DOTALL)
        if not fig_match:
            return []
        
        try:
            fig_json = fig_match.group(1)
            fig_data = json.loads(fig_json)
            
            if 'data' not in fig_data or not isinstance(fig_data['data'], list):
                return []
            
            components = []
            for item in fig_data['data']:
                component = self._parse_data_item(item)
                if component:
                    components.append(component)
            
            return components
        
        except json.JSONDecodeError:
            if self.debug:
                print("JSON parsing failed, trying to fix...")
            return self._try_fix_json_and_parse(fig_match.group(1))
    
    def _try_fix_json_and_parse(self, fig_json: str) -> List[MemoryComponent]:
        """Try to fix truncated JSON and parse again"""
        try:
            # Find last complete object
            last_complete = fig_json.rfind('"}')
            if last_complete != -1:
                fixed_json = fig_json[:last_complete+2] + ']}'
                fig_data = json.loads(fixed_json)
                
                if 'data' in fig_data:
                    components = []
                    for item in fig_data['data']:
                        component = self._parse_data_item(item)
                        if component:
                            components.append(component)
                    return components
        except:
            pass
        
        return []
    
    def _try_regex_parsing(self, script_content: str) -> List[MemoryComponent]:
        """Fallback regex parsing for extracting components"""
        if self.debug:
            print("Using regex parsing...")
        
        # Pattern for complete objects with x and y coordinates
        pattern = r'\{[^{}]*?"text":"([^"]+)"[^{}]*?"x":\[([^\]]+)\][^{}]*?"y":\[([^\]]+)\][^{}]*?"type":"scatter"[^{}]*?\}'
        matches = re.findall(pattern, script_content)
        
        components = []
        for match in matches:
            try:
                name = match[0]
                x_coords = [float(x.strip()) for x in match[1].split(',')]
                y_coords = [float(y.strip()) for y in match[2].split(',')]
                
                component = self._create_component_from_coords(name, x_coords, y_coords)
                if component:
                    components.append(component)
            
            except ValueError as e:
                if self.debug:
                    print(f"Error parsing coordinates for {match[0]}: {e}")
        
        return components
    
    def _parse_data_item(self, item: Dict) -> Optional[MemoryComponent]:
        """Parse a single data item into a MemoryComponent"""
        if not isinstance(item, dict) or 'text' not in item or 'y' not in item:
            return None
        
        name = item['text']
        y_coords = item['y']
        x_coords = item.get('x', [])
        
        if not isinstance(y_coords, list) or len(y_coords) < 2:
            return None
        
        return self._create_component_from_coords(name, x_coords, y_coords)
    
    def _create_component_from_coords(self, name: str, x_coords: List[float], y_coords: List[float]) -> Optional[MemoryComponent]:
        """Create a MemoryComponent from coordinate arrays"""
        if len(y_coords) < 2:
            return None
        
        y_min, y_max = min(y_coords), max(y_coords)
        memory_size = y_max - y_min
        
        if len(x_coords) >= 2:
            x_min, x_max = min(x_coords), max(x_coords)
            return MemoryComponent(
                name=name,
                memory_size=int(memory_size),
                y_start=int(y_min),
                y_end=int(y_max),
                x_start=x_min,
                x_end=x_max,
                x_coords=x_coords,
                y_coords=y_coords
            )
        else:
            return MemoryComponent(
                name=name,
                memory_size=int(memory_size),
                y_start=int(y_min),
                y_end=int(y_max),
                y_coords=y_coords
            )
    
    def identify_constant_parameters(self) -> Tuple[List[MemoryComponent], List[MemoryComponent]]:
        """Identify constant parameters (components alive throughout execution)"""
        # Find components with valid x coordinates
        valid_x_components = [c for c in self.components if c.x_start is not None and c.x_end is not None]
        
        if not valid_x_components:
            print("No components with x coordinates found")
            return [], self.components
        
        # Determine global x range
        self.global_x_min = min(c.x_start for c in valid_x_components)
        self.global_x_max = max(c.x_end for c in valid_x_components)
        
        print(f"Global X coordinate range: {self.global_x_min} to {self.global_x_max}")
        
        # Identify constant parameters
        const_params = []
        non_const_params = []
        tolerance = 0.1
        
        for component in self.components:
            if (component.x_start is not None and component.x_end is not None and
                abs(component.x_start - self.global_x_min) <= tolerance and
                abs(component.x_end - self.global_x_max) <= tolerance):
                
                component.is_constant = True
                # Classify waste type for constant parameters
                component.waste_type = self._classify_waste_type(component.name)
                const_params.append(component)
            else:
                non_const_params.append(component)
        
        print(f"Found {len(const_params)} constant parameters")
        print(f"Found {len(non_const_params)} temporary variables")
        
        return const_params, non_const_params
    
    def _classify_waste_type(self, name: str) -> str:
        """Classify the waste type based on component name"""
        name_lower = name.lower()
        
        if 'extract_const' in name_lower:
            return 'duplicate'
        elif 'inode' in name_lower:
            return 'bias'
        elif 'identity' in name_lower:
            return 'broadcast'
        else:
            return 'normal'
    
    def calculate_memory_breakdown(self, strategy_name: str = "") -> MemoryBreakdown:
        """Calculate comprehensive memory breakdown"""
        const_params, non_const_params = self.identify_constant_parameters()
        
        total_memory = sum(comp.memory_size for comp in self.components)
        constant_memory = sum(comp.memory_size for comp in const_params)
        activation_gradient_memory = total_memory - constant_memory
        
        # Get peak memory info with timing
        peak_memory, peak_time, peak_components = self.get_peak_memory_info()
        
        # Get gradient memory info at peak
        peak_gradient_memory, peak_gradient_count, peak_gradient_components = self.get_peak_gradient_memory_info()
        
        # Debug output to help diagnose the issue
        if self.debug:
            print(f"Debug: Creating MemoryBreakdown with peak_gradient_memory = {peak_gradient_memory}")
            print(f"Debug: peak_gradient_count = {peak_gradient_count}")
            print(f"Debug: peak_gradient_components length = {len(peak_gradient_components) if peak_gradient_components else 0}")
        
        # Create the breakdown object with all required fields
        breakdown = MemoryBreakdown(
            total_memory=total_memory,
            constant_memory=constant_memory,
            activation_gradient_memory=activation_gradient_memory,
            peak_memory_usage=peak_memory,
            strategy_name=strategy_name,
            peak_time=peak_time,
            peak_components=peak_components,
            peak_gradient_memory=peak_gradient_memory,
            peak_gradient_count=peak_gradient_count,
            peak_gradient_components=peak_gradient_components
        )
        
        if self.debug:
            print(f"Debug: Created breakdown object with attributes: {dir(breakdown)}")
            print(f"Debug: breakdown.peak_gradient_memory = {breakdown.peak_gradient_memory}")
        
        return breakdown
    
    def analyze(self, file_path: str) -> Tuple[List[MemoryComponent], List[MemoryComponent]]:
        """Main analysis method"""
        html_content = self.load_html_file(file_path)
        if not html_content:
            return [], []
        
        self.parse_html_content(html_content)
        return self.identify_constant_parameters()


class EnhancedMemoryReporter:
    """Enhanced reporter class for generating memory analysis reports"""
    
    @staticmethod
    def format_memory_size(size_bytes: int) -> str:
        """Format memory size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
        elif size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def print_enhanced_memory_summary(self, breakdown: MemoryBreakdown) -> None:
        """Print enhanced memory summary with all three categories"""
        print("\n" + "="*80)
        print("🧠 COMPREHENSIVE MEMORY BREAKDOWN")
        print("="*80)
        
        if breakdown.strategy_name:
            print(f"📋 Strategy: {breakdown.strategy_name}")
            print("-"*50)
        
        # Total Memory
        print(f"📊 TOTAL MEMORY: {self.format_memory_size(breakdown.total_memory)}")
        print(f"   Raw bytes: {breakdown.total_memory:,}")
        
        # Constant Memory (Parameters)
        const_percentage = (breakdown.constant_memory / breakdown.total_memory * 100) if breakdown.total_memory > 0 else 0
        print(f"\n🔧 CONSTANT MEMORY (Parameters): {self.format_memory_size(breakdown.constant_memory)}")
        print(f"   Raw bytes: {breakdown.constant_memory:,}")
        print(f"   Percentage: {const_percentage:.2f}%")
        
        # Activation + Gradient Memory
        act_grad_percentage = (breakdown.activation_gradient_memory / breakdown.total_memory * 100) if breakdown.total_memory > 0 else 0
        print(f"\n⚡ ACTIVATION + GRADIENT MEMORY: {self.format_memory_size(breakdown.activation_gradient_memory)}")
        print(f"   Raw bytes: {breakdown.activation_gradient_memory:,}")
        print(f"   Percentage: {act_grad_percentage:.2f}%")
        
        # Peak Memory Usage with Timing Information
        if breakdown.peak_memory_usage > 0:
            peak_percentage = (breakdown.peak_memory_usage / breakdown.total_memory * 100) if breakdown.total_memory > 0 else 0
            print(f"\n🔥 PEAK MEMORY USAGE: {self.format_memory_size(breakdown.peak_memory_usage)}")
            print(f"   Raw bytes: {breakdown.peak_memory_usage:,}")
            print(f"   Peak vs Total: {peak_percentage:.2f}%")
            
            # Show when peak occurs
            if breakdown.peak_time is not None:
                print(f"   ⏰ Peak occurs at time: {breakdown.peak_time:.4f}")
                
                # Show gradient memory statistics at peak
                if breakdown.peak_gradient_memory > 0:
                    gradient_peak_percentage = (breakdown.peak_gradient_memory / breakdown.peak_memory_usage * 100)
                    print(f"   🎯 GRADIENT/UPDATED MEMORY AT PEAK: {self.format_memory_size(breakdown.peak_gradient_memory)}")
                    print(f"      Raw bytes: {breakdown.peak_gradient_memory:,}")
                    print(f"      Gradient/Updated tensors count: {breakdown.peak_gradient_count:,}")
                    print(f"      Gradient/Updated memory / Peak memory: {gradient_peak_percentage:.2f}%")
                    
                    # Show some gradient component examples
                    if breakdown.peak_gradient_components:
                        print(f"      🔩 Gradient/Updated components at peak (showing first 5 of {len(breakdown.peak_gradient_components)}):")
                        display_grad_components = breakdown.peak_gradient_components[:5]
                        for i, comp_name in enumerate(display_grad_components, 1):
                            print(f"         {i}. {comp_name}")
                        if len(breakdown.peak_gradient_components) > 5:
                            print(f"         ... and {len(breakdown.peak_gradient_components) - 5} more gradient/updated components")
                else:
                    print(f"   🎯 GRADIENT/UPDATED MEMORY AT PEAK: 0 B (no gradient/updated tensors found)")
                
                # Show components active at peak
                if breakdown.peak_components:
                    non_gradient_count = len(breakdown.peak_components) - breakdown.peak_gradient_count
                    print(f"   🔩 All active components at peak ({len(breakdown.peak_components)} total: {breakdown.peak_gradient_count} gradients/updated + {non_gradient_count} others):")
                    
                    # Show first few components to avoid overwhelming output
                    display_components = breakdown.peak_components[:10]
                    for i, comp_name in enumerate(display_components, 1):
                        # Mark gradient/updated components with special indicator
                        comp_name_lower = comp_name.lower()
                        grad_indicator = " 🎯" if ('grad' in comp_name_lower or 'updated' in comp_name_lower) else ""
                        print(f"      {i:2d}. {comp_name}{grad_indicator}")
                    
                    if len(breakdown.peak_components) > 10:
                        print(f"      ... and {len(breakdown.peak_components) - 10} more components")
        
        # Memory Ratio Analysis
        if breakdown.constant_memory > 0 and breakdown.activation_gradient_memory > 0:
            ratio = breakdown.activation_gradient_memory / breakdown.constant_memory
            print(f"\n📈 MEMORY RATIOS:")
            print(f"   Activation+Gradient : Parameters = {ratio:.2f} : 1")
            print(f"   Parameters : Activation+Gradient = 1 : {ratio:.2f}")
            
            if ratio > 2:
                print(f"   💡 High activation/gradient memory - consider activation checkpointing")
            elif ratio < 0.5:
                print(f"   💡 Parameter-heavy model - consider parameter reduction techniques")
            else:
                print(f"   ✅ Balanced memory usage between parameters and activations")
        
        print("="*80)
    
    def save_memory_breakdown_csv(self, breakdowns: List[MemoryBreakdown], filename: str = "memory_breakdown.csv") -> None:
        """Save memory breakdown data to CSV"""
        if not breakdowns:
            print("No memory breakdown data to save")
            return
        
        # Prepare data for CSV
        csv_data = []
        for breakdown in breakdowns:
            total_mb = breakdown.total_memory / (1024 * 1024)
            const_mb = breakdown.constant_memory / (1024 * 1024)
            act_grad_mb = breakdown.activation_gradient_memory / (1024 * 1024)
            peak_mb = breakdown.peak_memory_usage / (1024 * 1024)
            
            const_pct = (breakdown.constant_memory / breakdown.total_memory * 100) if breakdown.total_memory > 0 else 0
            act_grad_pct = (breakdown.activation_gradient_memory / breakdown.total_memory * 100) if breakdown.total_memory > 0 else 0
            
            # Get gradient-related attributes
            peak_gradient_memory = breakdown.peak_gradient_memory
            peak_gradient_count = breakdown.peak_gradient_count
            peak_gradient_percentage = (peak_gradient_memory / breakdown.peak_memory_usage * 100) if breakdown.peak_memory_usage > 0 else 0
            
            csv_data.append({
                'strategy': breakdown.strategy_name or 'Unknown',
                'total_memory_mb': round(total_mb, 2),
                'constant_memory_mb': round(const_mb, 2),
                'activation_gradient_memory_mb': round(act_grad_mb, 2),
                'peak_memory_mb': round(peak_mb, 2),
                'peak_time': round(breakdown.peak_time, 4) if breakdown.peak_time is not None else None,
                'peak_components_count': len(breakdown.peak_components) if breakdown.peak_components else 0,
                'peak_gradient_memory_mb': round(peak_gradient_memory / (1024 * 1024), 2),
                'peak_gradient_memory_bytes': peak_gradient_memory,
                'peak_gradient_count': peak_gradient_count,
                'peak_gradient_percentage': round(peak_gradient_percentage, 2),
                'total_memory_bytes': breakdown.total_memory,
                'constant_memory_bytes': breakdown.constant_memory,
                'activation_gradient_memory_bytes': breakdown.activation_gradient_memory,
                'peak_memory_bytes': breakdown.peak_memory_usage,
                'constant_percentage': round(const_pct, 2),
                'activation_gradient_percentage': round(act_grad_pct, 2),
                'act_grad_to_const_ratio': round(act_grad_mb / const_mb, 2) if const_mb > 0 else 0
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        
        print(f"\n💾 Memory breakdown saved to: {filename}")
        print(f"📊 Contains {len(csv_data)} memory analysis entries")
        
        # Print CSV preview
        print(f"\n📋 CSV Preview:")
        print("-" * 100)
        print(df.to_string(index=False, max_cols=8))
        
        return df
    
    def print_const_memory_total(self, const_params: List[MemoryComponent]) -> None:
        """Print simple total of constant memory"""
        if not const_params:
            print("\n❌ No constant parameters found")
            return
        
        const_total_memory = sum(comp.memory_size for comp in const_params)
        
        print("\n" + "="*60)
        print("🎯 CONSTANT MEMORY TOTAL")
        print("="*60)
        print(f"📦 Number of constant parameters: {len(const_params):,}")
        print(f"💾 Total constant memory: {self.format_memory_size(const_total_memory)}")
        print(f"📊 Total constant memory (bytes): {const_total_memory:,}")
        
        # Add waste analysis
        self._print_const_waste_analysis(const_params)
        
        print("="*60)

    def _print_const_waste_analysis(self, const_params: List[MemoryComponent]) -> None:
        """Print constant memory waste analysis"""
        if not const_params:
            return
        
        # Categorize constants by waste type
        waste_categories = {
            'duplicate': [],    # EXTRACT_CONST
            'bias': [],         # inode (fusedmatmul bias)
            'broadcast': [],    # Identity (broadcast)
            'normal': []        # regular constants
        }
        
        for comp in const_params:
            waste_categories[comp.waste_type].append(comp)
        
        total_const_memory = sum(comp.memory_size for comp in const_params)
        
        print(f"\n🗂️ CONSTANT MEMORY WASTE BREAKDOWN:")
        print("-" * 55)
        
        # Calculate and display each category
        waste_stats = {}
        for category, components in waste_categories.items():
            if components:
                category_memory = sum(comp.memory_size for comp in components)
                category_pct = (category_memory / total_const_memory) * 100
                count_pct = (len(components) / len(const_params)) * 100
                
                waste_stats[category] = {
                    'count': len(components),
                    'memory': category_memory,
                    'memory_pct': category_pct,
                    'count_pct': count_pct
                }
        
        # Define category descriptions
        category_names = {
            'normal': '🟢 Normal Constants',
            'duplicate': '🔴 Duplicate Constants (EXTRACT_CONST)',
            'bias': '🟡 FusedMatMul Bias (inode)',
            'broadcast': '🟠 Broadcast Constants (Identity)'
        }
        
        # Print in order of priority (waste types first)
        order = ['duplicate', 'bias', 'broadcast', 'normal']
        
        for category in order:
            if category in waste_stats:
                stats = waste_stats[category]
                name = category_names[category]
                print(f"{name}:")
                print(f"  Count: {stats['count']:>4} ({stats['count_pct']:>5.1f}% of params)")
                print(f"  Memory: {self.format_memory_size(stats['memory']):>8} ({stats['memory_pct']:>5.1f}% of const memory)")
        
        # Calculate total waste
        total_waste_memory = sum(
            waste_stats[cat]['memory'] 
            for cat in ['duplicate', 'bias', 'broadcast'] 
            if cat in waste_stats
        )
        total_waste_count = sum(
            waste_stats[cat]['count'] 
            for cat in ['duplicate', 'bias', 'broadcast'] 
            if cat in waste_stats
        )
        
        if total_waste_memory > 0:
            waste_memory_pct = (total_waste_memory / total_const_memory) * 100
            waste_count_pct = (total_waste_count / len(const_params)) * 100
            
            print(f"\n💥 TOTAL WASTE SUMMARY:")
            print(f"  Wasted parameters: {total_waste_count:>4} ({waste_count_pct:>5.1f}%)")
            print(f"  Wasted memory: {self.format_memory_size(total_waste_memory):>8} ({waste_memory_pct:>5.1f}%)")
            print(f"  Potential savings: {self.format_memory_size(total_waste_memory)}")
        else:
            print(f"\n✅ No memory waste detected in constants!")


def extract_strategy_name_from_path(file_path: str) -> str:
    """Extract strategy name from file path"""
    # Try to extract from common path patterns
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Look for CCT strategy patterns
    for part in path_parts:
        if 'CCT' in part.upper():
            return part
    
    # Fallback to filename without extension
    filename = os.path.basename(file_path)
    return os.path.splitext(filename)[0]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced HTML memory analyzer with activation+gradient analysis, peak timing, and gradient/updated tensor tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_analyzer.py memory_alloc.html
  python enhanced_analyzer.py memory_alloc.html --debug
  python enhanced_analyzer.py memory_alloc.html --summary-only
  python enhanced_analyzer.py memory_alloc.html --strategy CCT_LA_LoRA
        """
    )
    
    parser.add_argument('file_path', help='Path to the HTML file to analyze')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--summary-only', action='store_true', help='Show only memory breakdown summary')
    parser.add_argument('--strategy', type=str, help='Strategy name for the analysis')
    parser.add_argument('--csv-output', type=str, default='memory_breakdown.csv', 
                       help='Output CSV filename (default: memory_breakdown.csv)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize analyzer and reporter
    analyzer = HTMLMemoryAnalyzer(debug=args.debug)
    reporter = EnhancedMemoryReporter()
    
    try:
        # Determine strategy name
        strategy_name = args.strategy or extract_strategy_name_from_path(args.file_path)
        
        # Perform analysis
        const_params, non_const_params = analyzer.analyze(args.file_path)
        breakdown = analyzer.calculate_memory_breakdown(strategy_name)
        
        # Generate reports
        if args.summary_only:
            # Only show enhanced summary
            reporter.print_enhanced_memory_summary(breakdown)
        else:
            # Show enhanced summary plus constant details
            reporter.print_enhanced_memory_summary(breakdown)
            reporter.print_const_memory_total(const_params)
        
        # Always save CSV
        reporter.save_memory_breakdown_csv([breakdown], args.csv_output)
        
        print(f"\n✅ Analysis completed for strategy: {strategy_name}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()