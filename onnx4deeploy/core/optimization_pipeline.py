# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Optimization Pipeline System for ONNX Models.

This module provides a flexible pipeline system for applying optimization passes
to ONNX models. Models can customize which optimizations to apply and in what order.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import onnx


@dataclass
class PassConfig:
    """Configuration for an optimization pass."""

    enabled: bool = True
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class OptimizationPass(ABC):
    """Base class for optimization passes."""

    def __init__(self, name: str, description: str = ""):
        """
        Initialize optimization pass.

        Args:
            name: Name of the optimization pass
            description: Human-readable description of what this pass does
        """
        self.name = name
        self.description = description

    @abstractmethod
    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        """
        Apply the optimization pass to an ONNX model.

        Args:
            onnx_file: Path to input ONNX file
            output_file: Path to save optimized ONNX file
            config: Configuration for this pass

        Returns:
            True if optimization was successful, False otherwise
        """

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"


class FunctionPass(OptimizationPass):
    """Optimization pass that wraps a simple function."""

    def __init__(
        self, name: str, func: Callable, description: str = "", requires_params: List[str] = None
    ):
        """
        Initialize function-based optimization pass.

        Args:
            name: Name of the optimization pass
            func: Function to call (signature: func(onnx_file, output_file, **kwargs))
            description: Human-readable description
            requires_params: List of required parameter names
        """
        super().__init__(name, description)
        self.func = func
        self.requires_params = requires_params or []

    def apply(self, onnx_file: str, output_file: str, config: PassConfig) -> bool:
        """Apply the function-based optimization pass."""
        try:
            # Check required parameters
            for param in self.requires_params:
                if param not in config.params:
                    raise ValueError(f"Pass '{self.name}' requires parameter '{param}'")

            # Call the function
            self.func(onnx_file, output_file, **config.params)
            return True
        except Exception as e:
            print(f"    ⚠️  Pass '{self.name}' failed: {e}")
            return False


class OptimizationPipeline:
    """
    Pipeline for applying a sequence of optimization passes to ONNX models.

    This class manages a queue of optimization passes and applies them in order.
    Models can customize which passes to use and configure each pass individually.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize optimization pipeline.

        Args:
            name: Name of this pipeline (e.g., "inference", "training")
        """
        self.name = name
        self.passes: List[OptimizationPass] = []
        self.pass_configs: Dict[str, PassConfig] = {}
        self.verbose = True

    def add_pass(
        self, optimization_pass: OptimizationPass, config: Optional[PassConfig] = None
    ) -> "OptimizationPipeline":
        """
        Add an optimization pass to the pipeline.

        Args:
            optimization_pass: The optimization pass to add
            config: Configuration for this pass (defaults to enabled with no params)

        Returns:
            Self for method chaining
        """
        self.passes.append(optimization_pass)
        self.pass_configs[optimization_pass.name] = config or PassConfig()
        return self

    def remove_pass(self, pass_name: str) -> "OptimizationPipeline":
        """
        Remove an optimization pass from the pipeline by name.

        Args:
            pass_name: Name of the pass to remove

        Returns:
            Self for method chaining
        """
        self.passes = [p for p in self.passes if p.name != pass_name]
        if pass_name in self.pass_configs:
            del self.pass_configs[pass_name]
        return self

    def configure_pass(self, pass_name: str, config: PassConfig) -> "OptimizationPipeline":
        """
        Update configuration for a specific pass.

        Args:
            pass_name: Name of the pass to configure
            config: New configuration for the pass

        Returns:
            Self for method chaining
        """
        if pass_name in self.pass_configs:
            self.pass_configs[pass_name] = config
        return self

    def disable_pass(self, pass_name: str) -> "OptimizationPipeline":
        """
        Disable a pass (keep in pipeline but skip execution).

        Args:
            pass_name: Name of the pass to disable

        Returns:
            Self for method chaining
        """
        if pass_name in self.pass_configs:
            self.pass_configs[pass_name].enabled = False
        return self

    def enable_pass(self, pass_name: str) -> "OptimizationPipeline":
        """
        Enable a previously disabled pass.

        Args:
            pass_name: Name of the pass to enable

        Returns:
            Self for method chaining
        """
        if pass_name in self.pass_configs:
            self.pass_configs[pass_name].enabled = True
        return self

    def run(self, onnx_file: str, output_file: str) -> Dict[str, bool]:
        """
        Run all enabled passes in the pipeline.

        Args:
            onnx_file: Path to input ONNX file
            output_file: Path to save final optimized ONNX file

        Returns:
            Dictionary mapping pass names to success status
        """
        if self.verbose:
            print(f"🔧 Running optimization pipeline: '{self.name}'")
            print(f"   Total passes: {len(self.passes)}")
            enabled_count = sum(1 for p in self.passes if self.pass_configs[p.name].enabled)
            print(f"   Enabled passes: {enabled_count}")

        results = {}
        current_file = onnx_file

        # Count original nodes
        if self.verbose:
            orig_model = onnx.load(onnx_file)
            orig_nodes = len(orig_model.graph.node)

        # Apply each pass
        for i, opt_pass in enumerate(self.passes):
            config = self.pass_configs[opt_pass.name]

            if not config.enabled:
                if self.verbose:
                    print(f"  ⊘ Skipping disabled pass: {opt_pass.name}")
                results[opt_pass.name] = False
                continue

            if self.verbose:
                print(f"  ➤ [{i+1}/{len(self.passes)}] {opt_pass.name}")
                if opt_pass.description:
                    print(f"      {opt_pass.description}")

            # For last pass, write to output_file; otherwise use temporary output
            next_file = output_file if i == len(self.passes) - 1 else current_file

            success = opt_pass.apply(current_file, next_file, config)
            results[opt_pass.name] = success

            if not success and self.verbose:
                print("      ⚠️  Pass failed (continuing with pipeline)")

        # Count final nodes
        if self.verbose:
            try:
                final_model = onnx.load(output_file)
                final_nodes = len(final_model.graph.node)
                print("\n  ✅ Pipeline complete")
                print(f"     Nodes: {orig_nodes} → {final_nodes} ({final_nodes - orig_nodes:+d})")
            except Exception:
                print("\n  ✅ Pipeline complete")

        return results

    def clone(self) -> "OptimizationPipeline":
        """
        Create a copy of this pipeline.

        Returns:
            New pipeline with same passes and configurations
        """
        new_pipeline = OptimizationPipeline(name=self.name)
        new_pipeline.passes = self.passes.copy()
        new_pipeline.pass_configs = {
            name: PassConfig(enabled=cfg.enabled, params=cfg.params.copy())
            for name, cfg in self.pass_configs.items()
        }
        new_pipeline.verbose = self.verbose
        return new_pipeline

    def __repr__(self):
        enabled = sum(1 for p in self.passes if self.pass_configs[p.name].enabled)
        return f"<OptimizationPipeline '{self.name}': {enabled}/{len(self.passes)} passes enabled>"
