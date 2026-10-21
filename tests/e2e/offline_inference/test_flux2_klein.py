# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""End-to-end tests for Flux2 Klein text-to-image and inpainting."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import DiffusionResponse, OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# Most Flux2 Klein tests have migrated to the common diffusion tests. This file
# retains behavior that requires the real model or offline client.
MODEL = "black-forest-labs/FLUX.2-klein-4B"

_OMNI_RUNNER_PARAM = (MODEL, None)

pytestmark = [pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)]

_HEIGHT = 512
_WIDTH = 512
_NUM_INFERENCE_STEPS = 4


def _create_test_image(width: int = _WIDTH, height: int = _HEIGHT, color: tuple = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _create_test_mask(width: int = _WIDTH, height: int = _HEIGHT) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([width // 4, height // 4, width * 3 // 4, height * 3 // 4], fill=255)
    return mask


def _create_test_inputs(color: tuple = (100, 150, 200)):
    return _create_test_image(_WIDTH, _HEIGHT, color), _create_test_mask(_WIDTH, _HEIGHT)


def _images_from_response(response: DiffusionResponse) -> list[Image.Image]:
    images = response.images
    assert images is not None
    return images


def _send_text2img_request(
    offline_client: OfflineOmniClient,
    prompt: str,
    seed: int,
    num_outputs_per_prompt: int = 1,
) -> DiffusionResponse:
    return offline_client.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": prompt,
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                seed=seed,
                num_outputs_per_prompt=num_outputs_per_prompt,
            ),
        }
    )


def _send_inpaint_with_generator(
    offline_client: OfflineOmniClient, prompt: str, input_image, mask_image, generator: torch.Generator
) -> DiffusionResponse:
    return offline_client.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": prompt,
            "multi_modal_data": {"image": input_image, "mask_image": mask_image},
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=generator,
                num_outputs_per_prompt=1,
            ),
        }
    )


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_text_to_image(offline_client: OfflineOmniClient):
    response = _send_text2img_request(offline_client, "A cat on a laptop", seed=42)

    images = _images_from_response(response)
    assert len(images) == 1
    assert_image_valid(images[0], width=_WIDTH, height=_HEIGHT)


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_text_to_image_deterministic(offline_client: OfflineOmniClient):
    prompt = "A mountain landscape at sunset"
    seed = 12345

    images1 = _images_from_response(_send_text2img_request(offline_client, prompt, seed=seed))
    images2 = _images_from_response(_send_text2img_request(offline_client, prompt, seed=seed))

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape
    assert np.array_equal(arr1, arr2), "Same prompt and seed should produce identical output"


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_text_to_image_different_seeds(offline_client: OfflineOmniClient):
    prompt = "A beautiful landscape"

    images1 = _images_from_response(_send_text2img_request(offline_client, prompt, seed=42))
    images2 = _images_from_response(_send_text2img_request(offline_client, prompt, seed=99999))

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape
    assert not np.array_equal(arr1, arr2), "Different seeds should produce different outputs"


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_text_to_image_multi_output(offline_client: OfflineOmniClient):
    response = _send_text2img_request(
        offline_client,
        "A red rose in a garden",
        seed=42,
        num_outputs_per_prompt=2,
    )

    images = _images_from_response(response)
    assert len(images) == 2
    for image in images:
        assert_image_valid(image, width=_WIDTH, height=_HEIGHT)
    arr0 = np.array(images[0])
    arr1 = np.array(images[1])
    assert arr0.shape == arr1.shape
    assert not np.array_equal(arr0, arr1), "Multi-output images should be different"


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_rejects_empty_prompt(offline_client: OfflineOmniClient):
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _send_text2img_request(offline_client, "", seed=42)


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_rejects_zero_inference_steps(offline_client: OfflineOmniClient):
    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        offline_client.send_diffusion_request(
            {
                "model": MODEL,
                "prompt": "A cat on a laptop",
                "sampling_params": OmniDiffusionSamplingParams(
                    height=_HEIGHT,
                    width=_WIDTH,
                    num_inference_steps=0,
                    seed=42,
                ),
            }
        )


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_handles_non_divisible_dimensions(offline_client: OfflineOmniClient):
    requested_size = 513
    response = offline_client.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "A cat on a laptop",
            "sampling_params": OmniDiffusionSamplingParams(
                height=requested_size,
                width=requested_size,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                seed=42,
            ),
        }
    )

    images = _images_from_response(response)
    assert len(images) == 1
    # Flux2 Klein rounds dimensions down to the nearest vae_scale_factor * 2.
    assert_image_valid(images[0], width=_WIDTH, height=_HEIGHT)


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_inpaint_deterministic(offline_client: OfflineOmniClient):
    input_image, mask_image = _create_test_inputs()
    seed = 12345

    gen1 = torch.Generator(current_omni_platform.device_type).manual_seed(seed)
    gen2 = torch.Generator(current_omni_platform.device_type).manual_seed(seed)

    images1 = _images_from_response(
        _send_inpaint_with_generator(offline_client, "A red flower in a field", input_image, mask_image, gen1)
    )
    images2 = _images_from_response(
        _send_inpaint_with_generator(offline_client, "A red flower in a field", input_image, mask_image, gen2)
    )

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape
    assert np.array_equal(arr1, arr2), "Same input and seed should produce identical output"


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_inpaint_different_seeds_different_output(offline_client: OfflineOmniClient):
    input_image, mask_image = _create_test_inputs()

    gen1 = torch.Generator(current_omni_platform.device_type).manual_seed(42)
    gen2 = torch.Generator(current_omni_platform.device_type).manual_seed(99999)

    images1 = _images_from_response(
        _send_inpaint_with_generator(offline_client, "A beautiful landscape", input_image, mask_image, gen1)
    )
    images2 = _images_from_response(
        _send_inpaint_with_generator(offline_client, "A beautiful landscape", input_image, mask_image, gen2)
    )

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape
    assert not np.array_equal(arr1, arr2), "Different seeds should produce different outputs"
