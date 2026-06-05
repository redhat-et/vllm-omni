# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline tests for Qwen3-TTS CustomVoice model with text input and audio output.

Async_chunk disable, cuda_graph disabled (no_async_chunk stage config).
CUDA graph is disabled by setting engine_args.enforce_eager=true via modify_stage_config().
Same structure as test_qwen3_omni (models, stage_configs, test_params, parametrize omni_runner).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def get_cuda_graph_config():
    """Build a temp deploy yaml mirroring the deleted qwen3_tts_no_async_chunk.yaml.

    Composes the synchronous (no-async-chunk) variant on top of the bundled
    qwen3_tts.yaml prod default, with cudagraphs disabled. Replaces the deleted
    standalone variant yaml; same effective config, no checked-in file needed.
    """
    return modify_stage_config(
        get_deploy_config_path("qwen3_tts.yaml"),
        updates={
            "async_chunk": False,
            "stages": {
                0: {
                    "max_num_seqs": 1,
                    "gpu_memory_utilization": 0.2,
                    "enforce_eager": True,
                    "async_scheduling": False,
                },
                1: {
                    "gpu_memory_utilization": 0.2,
                    "enforce_eager": True,
                    "async_scheduling": False,
                },
            },
        },
    )


# Same structure as test_qwen3_omni: models, stage_configs, test_params
tts_server_params = [
    pytest.param(
        (MODEL, get_cuda_graph_config()),
        id="no_cuda_graph",
    )
]


def get_prompt():
    """Text prompt for text-to-audio (same role as get_question in test_qwen3_omni)."""
    return "Hello, this is a test for text to audio."


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_runner, omni_runner_handler) -> None:
    """
    Test text input processing and audio output via offline Omni runner.
Input Modality: text
    Output Modality: audio
    Input Setting: stream=False
    Datasets: few requests
    """
    request_config = {"input": get_prompt(), "voice": "vivian"}
    omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_customvoice_special_characters(omni_runner, omni_runner_handler) -> None:
    """
    Test text-to-audio with heavy punctuation and special characters.
Input Modality: text with special characters (!,;$£€%...)
    Output Modality: audio
    """
    request_config = {
        "input": "Wait — really?! That costs $99.99; not £50 (or €45)... wow!",
        "voice": "vivian",
    }
    omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_customvoice_acronyms_and_abbreviations(omni_runner, omni_runner_handler) -> None:
    """
    Test text-to-audio with acronyms and abbreviations.
Input Modality: text with e.g., i.e., Dr., p.m., acronyms
    Output Modality: audio
    """
    request_config = {
        "input": "The CEO of NASA, e.g. Dr. Smith, will arrive at 3 p.m. i.e. before the Q&A.",
        "voice": "vivian",
    }
    omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_customvoice_chinese_text(omni_runner, omni_runner_handler) -> None:
    """
    Test text-to-audio with Chinese language text.
Input Modality: Chinese text
    Output Modality: audio
    Extra Setting: language=Chinese
    """
    request_config = {
        "input": "北京是中国的首都，有着悠久的历史和丰富的文化。",
        "voice": "vivian",
        "language": "Chinese",
    }
    omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_customvoice_rejects_empty_input(omni_runner, omni_runner_handler) -> None:
    """
    Negative test: empty input text should be rejected.
Input Modality: empty string
    Expected: ValueError
    """
    request_config = {"input": "", "voice": "vivian"}
    with pytest.raises(ValueError):
        omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
@pytest.mark.xfail(reason="Offline pipeline does not validate whitespace-only input (#3706)", strict=True)
def test_customvoice_rejects_whitespace_only_input(omni_runner, omni_runner_handler) -> None:
    """
    Negative test: whitespace-only input (spaces, tabs, newlines) should be rejected.
Input Modality: whitespace/tabs/newlines only
    Expected: ValueError
    """
    request_config = {"input": "   \t\n  ", "voice": "vivian"}
    with pytest.raises(ValueError):
        omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_customvoice_handles_very_long_input(omni_runner, omni_runner_handler) -> None:
    """
    Test text-to-audio with very long input text (~5000 chars).
Input Modality: long repeated text
    Output Modality: audio
    Extra Setting: max_new_tokens=512 to cap generation time
    """
    long_text = "This is a sentence that will be repeated many times to create a very long input. " * 60
    request_config = {
        "input": long_text,
        "voice": "vivian",
        "max_new_tokens": 512,
    }
    omni_runner_handler.send_audio_speech_request(request_config)
