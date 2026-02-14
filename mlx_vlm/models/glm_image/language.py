from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .config import ModelConfig, TextConfig


def _compute_default_rope_parameters(
    config: Optional[TextConfig] = None,
    **rope_kwargs,
) -> tuple[mx.array, float]:
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )

    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (mx.arange(0, dim, 2, dtype=mx.int64).astype(mx.float32) / dim)
    )
    return inv_freq, attention_factor


class GlmImageRotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.rope_type = config.rope_parameters.get("rope_type", "default")
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.mrope_section = config.rope_parameters.get("mrope_section", [8, 12, 12])

        self.rope_init_fn = _compute_default_rope_parameters
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self._inv_freq = mx.array(inv_freq, dtype=mx.float32)
        self._original_inv_freq = mx.array(inv_freq, dtype=mx.float32)

    def apply_mrope(self, freqs, mrope_section):
        split_indices = np.cumsum(mrope_section)[:-1].tolist()
        chunks = mx.split(freqs, split_indices, axis=-1)
        return mx.concatenate([chunk[i % 3] for i, chunk in enumerate(chunks)], axis=-1)

    def __call__(self, x, position_ids):
        inv_freq_expanded = self._inv_freq[None, None, :, None].astype(mx.float32)
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded, (3, position_ids.shape[1], self._inv_freq.shape[0], 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        freqs = (
            inv_freq_expanded.astype(mx.float32)
            @ position_ids_expanded.astype(mx.float32)
        ).transpose(0, 1, 3, 2)

        freqs = self.apply_mrope(freqs, self.mrope_section)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.flatten(mx.stack([-x2, x1], axis=-1), start_axis=-2, end_axis=-1)


def repeat_interleave(x, repeats, axis=-1):
    shape = list(x.shape)
    x = mx.expand_dims(x, axis=axis + 1 if axis >= 0 else axis)
    tile_shape = [1] * len(x.shape)
    tile_shape[axis + 1 if axis >= 0 else axis] = repeats
    x = mx.tile(x, tile_shape)
    new_shape = shape.copy()
    new_shape[axis] = shape[axis] * repeats
    return x.reshape(new_shape)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    cos = repeat_interleave(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
    sin = repeat_interleave(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


class GlmImageAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : k.shape[-2]]

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(out)


class GlmImageMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)


class GlmImageDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = GlmImageAttention(config)
        self.mlp = GlmImageMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x
        x = self.self_attn(self.input_layernorm(x), mask, cache, position_embeddings)
        x = self.post_self_attn_layernorm(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self.post_mlp_layernorm(x)
        return residual + x


class GlmImageTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GlmImageDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmImageRotaryEmbedding(config)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds.astype(self.norm.weight.dtype)

        if position_ids is None:
            offset = cache[0].offset if cache and cache[0] is not None else 0
            position_ids = mx.arange(offset, offset + h.shape[-2])
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))

        position_embeddings = self.rotary_emb(h, position_ids)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i], position_embeddings)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = GlmImageTextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vision_vocab_size, bias=False)
        self._rope_deltas = None
        self._position_ids = None

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        images_per_sample: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        batch_size, seq_length = input_ids.shape

        if image_grid_thw is None:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
            else:
                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )
            return position_ids.astype(input_ids.dtype), mx.zeros(
                (batch_size, 1), dtype=input_ids.dtype
            )

        grids_np = np.array(image_grid_thw)
        if images_per_sample is not None:
            split_sizes = np.array(images_per_sample).astype(np.int64).tolist()
            split_offsets = np.cumsum([0] + split_sizes)
            grids_per_sample = [
                grids_np[split_offsets[i] : split_offsets[i + 1]]
                for i in range(len(split_sizes))
            ]
        else:
            grids_per_sample = [grids_np] + [
                np.empty((0, 3), dtype=np.int64) for _ in range(batch_size - 1)
            ]

        image_start_token_id = self.config.image_start_token_id
        image_end_token_id = self.config.image_end_token_id

        pos_ids = np.ones((3, batch_size, seq_length), dtype=np.int64)

        for batch_idx in range(batch_size):
            curr_ids = np.array(input_ids[batch_idx]).astype(np.int64)
            if attention_mask is not None and attention_mask.shape[-1] == seq_length:
                valid_mask = np.array(attention_mask[batch_idx]) == 1
                valid_positions = np.where(valid_mask)[0]
                curr_valid_ids = curr_ids[valid_positions]
            else:
                valid_positions = np.arange(seq_length)
                curr_valid_ids = curr_ids

            starts = np.where(curr_valid_ids == image_start_token_id)[0] + 1
            ends = np.where(curr_valid_ids == image_end_token_id)[0]
            grids = (
                grids_per_sample[batch_idx]
                if batch_idx < len(grids_per_sample)
                else np.empty((0, 3), dtype=np.int64)
            )

            current_pos = 0
            prev_end = 0
            segments = []
            num_images = min(len(starts), len(ends), len(grids))

            for image_idx in range(num_images):
                start = int(starts[image_idx])
                end = int(ends[image_idx])
                if end < start:
                    continue

                text_len = max(start - prev_end, 0)
                if text_len > 0:
                    txt = np.arange(current_pos, current_pos + text_len, dtype=np.int64)
                    segments.append(np.stack([txt, txt, txt], axis=0))
                    current_pos += text_len

                _, h, w = grids[image_idx].tolist()
                image_seq_len = int(h * w)
                if image_seq_len > 0:
                    h_idx = np.repeat(np.arange(h, dtype=np.int64), w)
                    w_idx = np.tile(np.arange(w, dtype=np.int64), h)
                    temporal = np.full((image_seq_len,), current_pos, dtype=np.int64)
                    vision = np.stack(
                        [
                            temporal,
                            current_pos + h_idx,
                            current_pos + w_idx,
                        ],
                        axis=0,
                    )
                    segments.append(vision)
                    current_pos += int(max(h, w))

                prev_end = end

            remaining_len = max(len(curr_valid_ids) - prev_end, 0)
            if remaining_len > 0:
                txt = np.arange(
                    current_pos, current_pos + remaining_len, dtype=np.int64
                )
                segments.append(np.stack([txt, txt, txt], axis=0))

            if segments:
                curr_pos_ids = np.concatenate(segments, axis=1)
                expected_len = len(curr_valid_ids)
                if curr_pos_ids.shape[1] > expected_len:
                    curr_pos_ids = curr_pos_ids[:, :expected_len]
                elif curr_pos_ids.shape[1] < expected_len:
                    pad = np.repeat(
                        curr_pos_ids[:, -1:],
                        expected_len - curr_pos_ids.shape[1],
                        axis=1,
                    )
                    curr_pos_ids = np.concatenate([curr_pos_ids, pad], axis=1)
                pos_ids[:, batch_idx, valid_positions] = curr_pos_ids

        return mx.array(pos_ids, dtype=input_ids.dtype), mx.zeros(
            (batch_size, 1), dtype=input_ids.dtype
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        images_per_sample = kwargs.pop("images_per_sample", None)

        if position_ids is None:
            cache_offset = 0
            if cache and cache[0] is not None:
                offset = cache[0].offset
                cache_offset = (
                    offset.item() if isinstance(offset, mx.array) else int(offset)
                )

            if self._position_ids is not None:
                seq_len = inputs.shape[1]
                position_ids = self._position_ids[
                    :, :, cache_offset : cache_offset + seq_len
                ]
            else:
                position_ids, rope_deltas = self.get_rope_index(
                    inputs,
                    image_grid_thw=image_grid_thw,
                    images_per_sample=images_per_sample,
                    attention_mask=mask,
                )
                self._position_ids = position_ids
                self._rope_deltas = rope_deltas

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            mask=mask,
        )
        return LanguageModelOutput(logits=self.lm_head(out))

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
