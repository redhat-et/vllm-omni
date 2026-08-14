# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared env for ``http_invalid`` (real :func:`omni_server`)."""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image

from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from tests.helpers.mark import hardware_marks
from vllm_omni.entrypoints.utils import resolve_model_config_path
from vllm_omni.config.yaml_util import load_yaml_config

_SPEECH_SERVER_ARGS = ["--trust-remote-code", "--disable-log-stats"]
# Match ``tests/e2e/online_serving/*`` module-level env for subprocess serve.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_TEST_CLEAN_GPU_MEMORY", "0")

def pytest_addoption(parser):
    # TODO phrase better later filepath to list of models to validate and configuration
    parser.addoption("--model-validation-list", help="YAML list of files")

@pytest.fixture
def tiny_png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def pytest_generate_tests(metafunc):
    if "omni_server" in metafunc.fixturenames:
        if metafunc.config.getoption("model_validation_list"):
            config = load_yaml_config(metafunc.config.getoption("model_validation_list"))
            model_list = [] 
            for id, model in enumerate(config):
                param = pytest.param(
                    OmniServerParams(
                        model=model,
                        stage_config_path=resolve_model_config_path(model),
                        server_args=_SPEECH_SERVER_ARGS,
                    ),
                    id=str(id),
                    marks=hardware_marks(res={"cuda": "L4"}),
                )
                model_list.append(param)
        else:
            model_list = [
                pytest.param(
                    OmniServerParams(
                        model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
                        server_args=_SPEECH_SERVER_ARGS,
                    ),
                    id="qwen-tts",
                    marks=hardware_marks(res={"cuda": "L4"}),
                )
            ]
        metafunc.parametrize("omni_server", model_list, indirect=True)
        # Can use metafunc.cls.default_list to distinguish the default list based on the class
        # Create ABC with function default_list to be inherited for each class based on modality
        # Can follow similiar approach to get the default server args and marks for a modality
        # Could append class name to the id if it helps