# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from torchtitan.distributed.flex_shard.dist_muon import DistMuon


@dataclass(frozen=True)
class StorageParameter:
    device: torch.device

    def to_local(self):
        return SimpleNamespace(device=self.device)


@pytest.mark.parametrize(
    "device", ["cuda:0", f"{torch._C._get_privateuse1_backend_name()}:0"]
)
def test_storage_accepts_the_active_accelerator(device):
    parameter = StorageParameter(torch.device(device))
    optimizer = cast(DistMuon, SimpleNamespace(param_groups=[{"params": [parameter]}]))
    with (
        patch("torchtitan.distributed.flex_shard.dist_muon.DTensor", StorageParameter),
        patch("torch.accelerator.current_accelerator", return_value=parameter.device),
    ):
        assert DistMuon._validate_parameter_storage(optimizer) == parameter.device


@pytest.mark.parametrize(
    "devices, accelerator",
    [
        ([], "cuda"),
        (["cpu"], "cuda"),
        (["cuda:0", "cuda:1"], "cuda"),
        (["cuda:0"], None),
        (["cuda:0"], "cpu"),
    ],
)
def test_storage_rejects_invalid_devices(devices, accelerator):
    optimizer = cast(
        DistMuon,
        SimpleNamespace(
            param_groups=[
                {"params": [StorageParameter(torch.device(device)) for device in devices]}
            ]
        ),
    )
    with (
        patch("torchtitan.distributed.flex_shard.dist_muon.DTensor", StorageParameter),
        patch(
            "torch.accelerator.current_accelerator",
            return_value=None if accelerator is None else torch.device(accelerator),
        ),
        pytest.raises(ValueError, match="one accelerator device"),
    ):
        DistMuon._validate_parameter_storage(optimizer)
