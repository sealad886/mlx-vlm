from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def _check_array_shape(arr):
    shape = arr.shape
    if len(shape) == 4:
        out_channels, k_h, k_w, _ = shape
        return (out_channels >= k_h) and (out_channels >= k_w) and (k_h == k_w)
    return False


class GlmImageVisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GlmImageVisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.qkv = nn.Linear(
            config.hidden_size, config.hidden_size * 3, bias=config.attention_bias
        )
        self.proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(self, hidden_states: mx.array, cu_seqlens: mx.array) -> mx.array:
        seq_length = hidden_states.shape[0]

        qkv = self.qkv(hidden_states).reshape(
            seq_length, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.transpose(1, 0, 2, 3)
        q = qkv[0].transpose(1, 0, 2)[None, ...]
        k = qkv[1].transpose(1, 0, 2)[None, ...]
        v = qkv[2].transpose(1, 0, 2)[None, ...]

        lengths: Sequence[int] = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_indices = []
        cumsum = 0
        for length in lengths[:-1]:
            cumsum += int(length)
            split_indices.append(cumsum)

        q_splits = mx.split(q, split_indices, axis=2)
        k_splits = mx.split(k, split_indices, axis=2)
        v_splits = mx.split(v, split_indices, axis=2)

        outputs = []
        for q_chunk, k_chunk, v_chunk in zip(q_splits, k_splits, v_splits):
            out = mx.fast.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                scale=self.scaling,
                mask=None,
            )
            outputs.append(out)

        attn_output = mx.concatenate(outputs, axis=2)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class GlmImageVisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 3)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, self.embed_dim)


class GlmImageVisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.hidden_size)

    def __call__(
        self,
        embeddings: mx.array,
        lengths: List[int],
        image_shapes: mx.array,
        h_coords: mx.array,
        w_coords: mx.array,
    ) -> mx.array:
        base_size = int(self.num_patches**0.5)

        target_h = []
        target_w = []
        image_shapes_np = np.array(image_shapes)
        for idx, length in enumerate(lengths):
            target_h.extend([int(image_shapes_np[idx, 1])] * int(length))
            target_w.extend([int(image_shapes_np[idx, 2])] * int(length))

        h_coords_np = np.array(h_coords, dtype=np.float32)
        w_coords_np = np.array(w_coords, dtype=np.float32)
        target_h_np = np.array(target_h, dtype=np.float32)
        target_w_np = np.array(target_w, dtype=np.float32)

        h_idx = np.floor(
            ((h_coords_np + 0.5) / np.maximum(target_h_np, 1.0)) * base_size
        )
        w_idx = np.floor(
            ((w_coords_np + 0.5) / np.maximum(target_w_np, 1.0)) * base_size
        )
        h_idx = np.clip(h_idx.astype(np.int32), 0, base_size - 1)
        w_idx = np.clip(w_idx.astype(np.int32), 0, base_size - 1)

        pos_ids = mx.array(h_idx * base_size + w_idx, dtype=mx.int32)
        pos_embeds = self.position_embedding(pos_ids)
        return embeddings + pos_embeds


class GlmImageVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GlmImageVisionAttention(config)
        self.mlp = GlmImageVisionMLP(config)

    def __call__(self, hidden_states: mx.array, cu_seqlens: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = GlmImageVisionEmbeddings(config)
        self.patch_embed = GlmImageVisionPatchEmbed(config)
        self.blocks = [GlmImageVisionBlock(config) for _ in range(config.depth)]

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        pos_ids = []
        for t, h, w in np.array(grid_thw).tolist():
            hpos_ids = np.arange(h)[:, None].repeat(w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.transpose(0, 2, 1, 3).reshape(-1)

            wpos_ids = np.arange(w)[None, :].repeat(h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.transpose(0, 2, 1, 3).reshape(-1)

            stacked = np.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(np.tile(stacked, (t, 1)))

        return mx.array(np.concatenate(pos_ids, axis=0), dtype=mx.int32)

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        hidden_states = self.patch_embed(pixel_values)
        image_type_ids = self.rot_pos_emb(grid_thw)

        seq_lens = np.array(grid_thw[:, 1] * grid_thw[:, 2]).astype(np.int64)
        repeats = np.array(grid_thw[:, 0]).astype(np.int64)
        repeated_values = []
        for seq_len, repeat_count in zip(seq_lens.tolist(), repeats.tolist()):
            repeated_values.extend([int(seq_len)] * int(repeat_count))

        cu_seqlens = np.cumsum(repeated_values)
        cu_seqlens = np.pad(cu_seqlens, (1, 0), constant_values=0)
        cu_seqlens = mx.array(cu_seqlens, dtype=mx.int32)

        hidden_states = self.embeddings(
            hidden_states,
            repeated_values,
            image_shapes=grid_thw,
            h_coords=image_type_ids[:, 0],
            w_coords=image_type_ids[:, 1],
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens)

        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if "patch_embed.proj.weight" in key:
                if _check_array_shape(value):
                    sanitized_weights[key] = value
                else:
                    sanitized_weights[key] = value.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[key] = value
        return sanitized_weights
