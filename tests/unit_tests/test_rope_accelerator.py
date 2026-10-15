# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.tools.utils import device_module, device_type


@pytest.mark.parametrize("on_accelerator", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("position_mode", ["prefix", "singleton", "packed"])
def test_complex_rope_matches_cpu(
    on_accelerator: bool, dtype: torch.dtype, position_mode: str
):
    if on_accelerator and not device_module.is_available():
        pytest.skip("An accelerator is unavailable")
    device = device_type if on_accelerator else "cpu"
    torch.manual_seed(42)
    reference = ComplexRoPE.Config(dim=16, max_seq_len=32).build()
    candidate = deepcopy(reference).to(device)
    query = torch.randn(2, 8, 4, 16, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 8, 2, 16, dtype=dtype, requires_grad=True)
    positions = None
    if position_mode == "singleton":
        positions = torch.tensor([[7, 2, 5, 1, 0, 3, 4, 6]])
    elif position_mode == "packed":
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4], [2, 0, 1, 0, 1, 2, 0, 1]])
    device_query = query.detach().to(device).clone().requires_grad_(True)
    device_key = key.detach().to(device).clone().requires_grad_(True)
    device_positions = positions.to(device) if positions is not None else None
    expected = reference(query, key, positions)
    actual = candidate(device_query, device_key, device_positions)
    for output, ref_output in zip(actual, expected):
        torch.testing.assert_close(output.cpu(), ref_output)
    (expected[0].float().sum() + expected[1].float().sum()).backward()
    (actual[0].float().sum() + actual[1].float().sum()).backward()
    assert device_query.grad is not None and query.grad is not None
    assert device_key.grad is not None and key.grad is not None
    torch.testing.assert_close(device_query.grad.cpu(), query.grad)
    torch.testing.assert_close(device_key.grad.cpu(), key.grad)
