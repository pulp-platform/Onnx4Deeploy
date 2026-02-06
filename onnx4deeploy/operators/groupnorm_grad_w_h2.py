# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""GroupNormGradWH2 operator test implementation (for H=2 cases)."""

from typing import Any, Dict

import numpy as np

from .groupnorm_grad_w import GroupNormGradWOperatorTest


class GroupNormGradWH2OperatorTest(GroupNormGradWOperatorTest):
    """Test generator for GroupNormGradW with H=2 (like EPIDENET layer_norm3)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = (1, 16, 2, 20)  # Default H=2

    def get_operator_name(self) -> str:
        return "GroupNormGradWH2"

    def load_config(self) -> Dict[str, Any]:
        """Load GroupNormGradWH2-specific configuration."""
        config = super().load_config()

        # Try to find config under different keys
        gngw_config = config.get("groupnormgradwh2", config.get("groupnormgradw", {}))
        self.input_shape = tuple(gngw_config.get("input_shape", [1, 16, 2, 20]))
        self.num_groups = gngw_config.get("num_groups", 1)
        self.epsilon = gngw_config.get("epsilon", 0.001)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate inputs with fixed seed for reproducibility."""
        # Use fixed seed like the legacy test
        np.random.seed(42)
        return super().generate_inputs()
