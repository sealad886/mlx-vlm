from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen3_5.qwen3_5 import Model as DenseModel
from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model as MoEModel


def _model_for_sanitize(model_class):
    model = model_class.__new__(model_class)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            tie_word_embeddings=False,
            num_hidden_layers=0,
            mtp_num_hidden_layers=0,
        )
    )
    return model


def _weights_with_draft_shard():
    return {
        "language_model.model.layers.0.input_layernorm.weight": mx.array([2.0]),
        "language_model.mtp.layers.0.input_layernorm.weight": mx.array([1.0]),
    }


def test_dense_qwen_mtp_shard_does_not_shift_base_norm_weights():
    weights = _model_for_sanitize(DenseModel).sanitize(_weights_with_draft_shard())

    assert weights["language_model.model.layers.0.input_layernorm.weight"].tolist() == [
        2.0
    ]
    assert not any("mtp." in key for key in weights)


def test_moe_qwen_mtp_shard_does_not_shift_base_norm_weights():
    weights = _model_for_sanitize(MoEModel).sanitize(_weights_with_draft_shard())

    assert weights["language_model.model.layers.0.input_layernorm.weight"].tolist() == [
        2.0
    ]
    assert not any("mtp." in key for key in weights)


@pytest.mark.parametrize("model_class", [DenseModel, MoEModel])
@pytest.mark.parametrize("raw_prefix", ["mtp.", "model.language_model.mtp."])
def test_integrated_mtp_raw_norms_convert_once(model_class, raw_prefix):
    model = _model_for_sanitize(model_class)
    model.config.text_config.mtp_num_hidden_layers = 1
    suffixes = (
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
    )
    raw = {raw_prefix + suffix: mx.zeros((8,)) for suffix in suffixes}
    converted = model.sanitize(raw)
    reloaded = model.sanitize(dict(converted))

    for suffix in suffixes:
        key = "language_model.mtp." + suffix
        assert mx.array_equal(converted[key], mx.ones((8,))).item()
        assert mx.array_equal(reloaded[key], converted[key]).item()


@pytest.mark.parametrize("model_class", [DenseModel, MoEModel])
def test_integrated_mtp_converted_norms_preserve_values(model_class):
    model = _model_for_sanitize(model_class)
    model.config.text_config.mtp_num_hidden_layers = 1
    weights = {
        "language_model.mtp.pre_fc_norm_embedding.weight": mx.array([2.0]),
        "language_model.mtp.pre_fc_norm_hidden.weight": mx.array([3.0]),
        "language_model.mtp.norm.weight": mx.array([4.0]),
        "language_model.model.norm.weight": mx.array([5.0]),
    }
    sanitized = model.sanitize(dict(weights))
    for key, value in weights.items():
        assert mx.array_equal(sanitized[key], value).item()


@pytest.mark.parametrize(
    "namespace", ["mtp", "model.language_model.mtp", "language_model.mtp"]
)
def test_moe_mtp_combined_experts_convert_in_each_namespace(namespace):
    model = _model_for_sanitize(MoEModel)
    model.config.text_config.mtp_num_hidden_layers = 1
    prefix = namespace + ".layers.0.mlp"
    combined = mx.arange(2 * 6 * 4).reshape(2, 6, 4)
    down = mx.arange(2 * 4 * 3).reshape(2, 4, 3)
    weights = model.sanitize(
        {
            prefix + ".experts.gate_up_proj": combined,
            prefix + ".experts.down_proj": down,
        }
    )
    target = "language_model.mtp.layers.0.mlp.switch_mlp"
    assert mx.array_equal(weights[target + ".gate_proj.weight"], combined[:, :3]).item()
    assert mx.array_equal(weights[target + ".up_proj.weight"], combined[:, 3:]).item()
    assert mx.array_equal(weights[target + ".down_proj.weight"], down).item()
    reloaded = model.sanitize(dict(weights))
    assert set(reloaded) == set(weights)
    for key in weights:
        assert mx.array_equal(reloaded[key], weights[key]).item()
