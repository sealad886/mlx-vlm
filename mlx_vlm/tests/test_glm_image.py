"""GLM-Image encoder, decoding and composite-checkpoint contracts."""

import json
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.models import glm_image
from mlx_vlm.models.cache import KVCache
from mlx_vlm.utils import (
    find_model_config_path,
    load_config,
    load_model,
    load_processor,
    prepare_inputs,
    resolve_processor_directory,
    save_config,
    save_weights,
)


def tiny_config():
    return glm_image.ModelConfig(
        text_config=glm_image.TextConfig(
            vocab_size=64,
            vision_vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            rope_parameters={
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000,
            },
        ),
        vision_config=glm_image.VisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            image_size=4,
        ),
        vq_config=glm_image.VQConfig(
            embed_dim=16,
            num_embeddings=16,
            latent_channels=16,
        ),
        image_token_id=63,
        image_start_token_id=61,
        image_end_token_id=62,
    )


def config_dict(config):
    result = config.to_dict()
    for name in ("text_config", "vision_config", "vq_config"):
        result[name] = getattr(config, name).to_dict()
    return result


def test_hugging_face_configuration_loads_nested_metadata():
    from transformers.models.glm_image.configuration_glm_image import GlmImageConfig

    exported = GlmImageConfig().to_dict()
    config = glm_image.ModelConfig.from_dict(exported)
    for field in ("text_config", "vision_config", "vq_config"):
        assert getattr(config, field).model_type == exported[field]["model_type"]
        assert getattr(config, field).to_dict()
    assert config.text_config.hidden_size == exported["text_config"]["hidden_size"]
    assert config.vision_config.image_size == exported["vision_config"]["image_size"]
    assert config.vq_config.num_embeddings == exported["vq_config"]["num_embeddings"]


def test_vision_attention_preserves_head_channels_and_image_boundaries():
    from mlx_vlm.models.glm_image.vision import GlmImageVisionAttention

    config = glm_image.VisionConfig(hidden_size=4, num_heads=2)
    attention = GlmImageVisionAttention(config)
    # Zero queries/keys produce uniform within-image attention. Identity value
    # and output projections expose channel ordering without a random oracle.
    attention.qkv.weight = mx.concatenate([mx.zeros((8, 4)), mx.eye(4)])
    attention.qkv.bias = mx.zeros((12,))
    attention.proj.weight = mx.eye(4)
    attention.proj.bias = mx.zeros((4,))
    values = mx.array(
        [
            [1, 10, 100, 1000],
            [3, 30, 300, 3000],
            [5, 50, 500, 5000],
            [7, 70, 700, 7000],
            [9, 90, 900, 9000],
        ],
        dtype=mx.float32,
    )
    actual = attention(values, mx.array([0, 2, 5]))
    expected = [[2, 20, 200, 2000]] * 2 + [[7, 70, 700, 7000]] * 3
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_vision_positions_interpolate_with_half_pixel_border_coordinates(dtype):
    from mlx_vlm.models.glm_image.vision import GlmImageVisionEmbeddings

    embedding = GlmImageVisionEmbeddings(
        glm_image.VisionConfig(hidden_size=2, image_size=4, patch_size=2)
    )
    embedding.position_embedding.weight = mx.array(
        [[0, 0], [2, 4], [10, 20], [12, 24]], dtype=dtype
    )
    # Sample a nonsquare 3x4 grid in scrambled patch order plus a second 1x1
    # image. These hand-computed values cover interior blending and all borders.
    h = mx.array([2, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
    w = mx.array([3, 0, 2, 1, 0, 3, 3, 1, 0, 2, 2, 1, 0])
    actual = embedding(
        mx.zeros((13, 2), dtype=dtype), [12, 1], mx.array([[1, 3, 4], [1, 1, 1]]), h, w
    )
    expected = np.array([12, 0, 6.5, 0.5, 10, 7, 2, 10.5, 5, 1.5, 11.5, 5.5, 6])
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.stack([expected, expected * 2], axis=-1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert actual.dtype == dtype


@pytest.mark.parametrize("head_dim", [4, 6])
def test_rotary_embedding_matches_checkpoint_half_channel_layout(head_dim):
    from mlx_vlm.models.glm_image.language import apply_rotary_pos_emb

    q = np.arange(1, 1 + 2 * 2 * head_dim, dtype=np.float32).reshape(1, 2, 2, head_dim)
    k = -q[:, :1]
    angles = np.array([[0.4, 0.8, 0.4, 0.8], [0.2, 0.6, 0.2, 0.6]], dtype=np.float32)
    cos, sin = np.cos(angles)[None], np.sin(angles)[None]
    actual_q, actual_k = apply_rotary_pos_emb(
        mx.array(q), mx.array(k), mx.array(cos), mx.array(sin)
    )
    for actual, original in ((actual_q, q), (actual_k, k)):
        rotated = np.concatenate([-original[..., 2:4], original[..., :2]], axis=-1)
        expected = np.concatenate(
            [
                original[..., :4] * cos[:, None] + rotated * sin[:, None],
                original[..., 4:],
            ],
            axis=-1,
        )
        np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=1e-6)


def test_text_generation_continues_after_prefill():
    model = glm_image.Model(tiny_config())
    cache = [KVCache()]
    inputs = mx.array([[1, 2, 3]])
    embeds = model.get_input_embeddings(inputs).inputs_embeds
    first = model.language_model(inputs, inputs_embeds=embeds, cache=cache).logits
    second = model.language_model(mx.array([[4]]), cache=cache).logits
    third = model.language_model(mx.array([[5]]), cache=cache).logits
    mx.eval(first, second, third)
    assert first.shape == (1, 3, 32)
    assert second.shape == third.shape == (1, 1, 32)
    assert cache[0].offset == 5
    assert bool(mx.all(mx.isfinite(third)))


def test_target_grid_positions_continue_in_spatial_order():
    model = glm_image.Model(tiny_config())
    inputs = mx.array([[1, 2, 61]])
    grids = mx.array([[1, 2, 3], [1, 1, 2]])
    features = model.get_input_embeddings(
        inputs, image_grid_thw=grids, images_per_sample=mx.array([2])
    )
    cache = [KVCache()]
    logits = model.language_model(
        inputs, inputs_embeds=features.inputs_embeds, cache=cache
    ).logits
    # The low-resolution target is emitted first, as in Transformers GLM-Image.
    for _ in range(4):
        logits = model.language_model(mx.array([[4]]), cache=cache).logits
        mx.eval(logits)
        assert logits.shape == (1, 1, 32)
    expected = np.array([[3, 3, 5, 5], [3, 3, 5, 5], [3, 4, 5, 6]])
    np.testing.assert_array_equal(
        np.array(model.language_model._decode_position_ids[:, 0, :4]), expected
    )


def test_public_token_generator_prefills_and_decodes_target_grids():
    from mlx_vlm.generate import generate_step

    model = glm_image.Model(tiny_config())
    # Match the test runner's chosen backend; production's generation stream is
    # left unchanged. This permits CPU proof when native Metal is unavailable.
    with patch(
        "mlx_vlm.generate.ar.generation_stream", mx.default_stream(mx.default_device())
    ):
        outputs = list(
            generate_step(
                mx.array([[1, 2, 61]]),
                model,
                None,
                mx.ones((1, 3), dtype=mx.int32),
                image_grid_thw=mx.array([[1, 2, 3], [1, 1, 2]]),
                images_per_sample=mx.array([2]),
                max_tokens=4,
                temperature=0,
            )
        )
    assert len(outputs) == 4
    assert all(0 <= int(token) < 32 for token, _ in outputs)


def test_image_tokenization_and_placeholder_validation():
    model = glm_image.Model(tiny_config())
    pixels = mx.ones((4, 3 * 2 * 2))
    grids = mx.array([[1, 2, 2], [1, 2, 3]])
    inputs = mx.array([[1, 61, 63, 63, 63, 63, 62, 2, 61]])
    features = model.get_input_embeddings(
        inputs,
        pixels,
        image_grid_thw=grids,
        images_per_sample=mx.array([2]),
    )
    mx.eval(features.inputs_embeds)
    assert features.inputs_embeds.shape == (1, 9, 16)
    assert bool(mx.all(mx.isfinite(features.inputs_embeds)))
    with pytest.raises(ValueError, match="placeholder"):
        model.get_input_embeddings(
            mx.array([[1, 61, 63, 62, 2, 61]]),
            pixels,
            image_grid_thw=grids,
            images_per_sample=mx.array([2]),
        )


class Tokenizer:
    image_token = "<|image|>"
    grid_bos_token = "<sop>"
    grid_eos_token = "<eop>"
    bos_token = "<bos>"
    pad_token = "<pad>"
    eos_token = "<eos>"
    model_input_names = ["input_ids", "attention_mask"]

    def __call__(self, text, **kwargs):
        self.last_texts = list(text) if isinstance(text, list) else [text]
        return {
            "input_ids": [[1, 2, 3]] * len(self.last_texts),
            "attention_mask": [[1, 1, 1]] * len(self.last_texts),
        }

    def convert_tokens_to_ids(self, token):
        return 63 if token == self.image_token else 0


class ImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __call__(self, images, **kwargs):
        return {
            "pixel_values": np.zeros((len(images) * 4, 12), dtype=np.float32),
            "image_grid_thw": np.array([[1, 2, 2]] * len(images)),
        }


def processor_fixture():
    # Only tokenization/processor behavior is under test; avoid remote resources.
    processor = glm_image.GlmImageProcessor.__new__(glm_image.GlmImageProcessor)
    processor.tokenizer = Tokenizer()
    processor.image_processor = ImageProcessor()
    for name in ("image_token", "grid_bos_token", "grid_eos_token", "bos_token"):
        setattr(processor, name, getattr(processor.tokenizer, name))
    processor.image_token_id = 63
    return processor


def test_prepare_inputs_runs_processor_for_text_to_image():
    processor = processor_fixture()
    inputs = prepare_inputs(
        processor, prompts=["draw a bird"], target_h=64, target_w=96
    )
    assert inputs["input_ids"].shape == (1, 3)
    np.testing.assert_array_equal(inputs["image_grid_thw"][0], [1, 2, 3])
    np.testing.assert_array_equal(inputs["images_per_sample"], [2])
    assert processor.tokenizer.last_texts[0] != "draw a bird"


def test_prepare_inputs_expands_source_image_slots():
    processor = processor_fixture()
    inputs = prepare_inputs(
        processor,
        prompts=["edit <|image|>"],
        images=[Image.new("RGB", (4, 4))],
        target_h=64,
        target_w=96,
    )
    assert processor.tokenizer.last_texts[0].count("<|image|>") == 4
    np.testing.assert_array_equal(inputs["image_grid_thw"], [[1, 2, 2], [1, 2, 3]])
    assert inputs["pixel_values"].shape == (4, 12)


def test_native_image_processor_normalizes_and_patchifies_source_grid():
    processor = glm_image.GlmImageImageProcessor(
        patch_size=2,
        min_pixels=32 * 32,
        max_pixels=32 * 32,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )
    outputs = processor([Image.new("RGB", (32, 32), color=(0, 255, 0))])
    np.testing.assert_array_equal(outputs["image_grid_thw"], [[1, 8, 8]])
    assert outputs["pixel_values"].shape == (64, 12)
    np.testing.assert_array_equal(
        outputs["pixel_values"][0], [0] * 4 + [1] * 4 + [0] * 4
    )


def test_real_processor_saves_and_loads_without_torch(tmp_path):
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=FastTokenizer(
            WordLevel({"[UNK]": 0, "<eos>": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    processor = glm_image.GlmImageProcessor(
        tokenizer=tokenizer,
        image_processor=glm_image.GlmImageImageProcessor(patch_size=2),
        chat_template="{{ messages[0]['content'] }}",
    )
    processor.save_pretrained(tmp_path)
    save_config(config_dict(tiny_config()), tmp_path / "config.json")
    restored = load_processor(tmp_path, add_detokenizer=False)
    assert isinstance(restored.image_processor, glm_image.GlmImageImageProcessor)
    inputs = prepare_inputs(restored, prompts=["draw a bird"], target_h=64, target_w=96)
    np.testing.assert_array_equal(inputs["image_grid_thw"][0], [1, 2, 3])


def write_snapshot(root):
    model_dir = root / "vision_language_encoder"
    model_dir.mkdir(parents=True)
    config = tiny_config()
    save_config(config_dict(config), model_dir / "config.json")
    (model_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": 31}))
    save_weights(model_dir, glm_image.Model(config))
    processor_dir = root / "processor"
    processor_dir.mkdir()
    (processor_dir / "tokenizer_config.json").write_text("{}")
    (processor_dir / "chat_template.jinja").write_text("{{ messages }}")
    return model_dir, processor_dir


def test_nested_checkpoint_loads_and_preserves_generated_weight_layout(tmp_path):
    model_dir, processor_dir = write_snapshot(tmp_path)
    assert find_model_config_path(tmp_path) == model_dir / "config.json"
    assert resolve_processor_directory(tmp_path) == processor_dir
    assert load_config(tmp_path)["eos_token_id"] == 31
    model = load_model(tmp_path)
    logits = model.language_model(mx.array([[1, 2]])).logits
    mx.eval(logits)
    assert logits.shape == (1, 2, 32)
    with patch.object(
        glm_image.GlmImageProcessor, "from_pretrained", return_value="processor"
    ):
        assert load_processor(tmp_path, add_detokenizer=False) == "processor"


def test_root_config_wins_and_ambiguous_nested_models_fail(tmp_path):
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text('{"model_type":"glm_image"}')
    with pytest.raises(ValueError, match="Ambiguous"):
        find_model_config_path(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"glm_image"}')
    assert find_model_config_path(tmp_path) == tmp_path / "config.json"


def test_conversion_retains_nested_metadata_and_its_own_shard_index(tmp_path):
    import importlib

    convert_module = importlib.import_module("mlx_vlm.convert")

    source = tmp_path / "source"
    model_dir, _ = write_snapshot(source)
    (model_dir / "model.safetensors.index.json").write_text('{"old_index":true}')
    (model_dir / "custom.py").write_text("value = 1\n")
    output = tmp_path / "converted"
    model = glm_image.Model(tiny_config())
    with (
        patch.object(
            convert_module,
            "fetch_from_hub",
            return_value=(model, config_dict(model.config), object()),
        ),
        patch.object(convert_module, "create_model_card"),
    ):
        convert_module.convert(str(source), str(output), dtype="float32")
    assert json.loads((output / "config.json").read_text())["model_type"] == "glm_image"
    assert "weight_map" in json.loads(
        (output / "model.safetensors.index.json").read_text()
    )
    assert (output / "custom.py").read_text() == "value = 1\n"
    assert (output / "chat_template.jinja").is_file()
    assert (output / "tokenizer_config.json").is_file()
    assert not (output / "vision_language_encoder").exists()
    reloaded = load_model(output)
    logits = reloaded.language_model(mx.array([[1, 2]])).logits
    mx.eval(logits)
    assert logits.shape == (1, 2, 32)
