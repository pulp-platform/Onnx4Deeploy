# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT


def register_model(func):
    """
    Fallback wrapper in case timm isn't installed
    """
    return func
