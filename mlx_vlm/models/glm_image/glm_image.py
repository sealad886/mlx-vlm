from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig, VQConfig
from .language import LanguageModel
from .vision import VisionModel


def _compute_split_indices(split_sizes):
    split_indices = []
    cumsum = 0
    for size in split_sizes[:-1]:
        cumsum += int(size)
        split_indices.append(cumsum)
    return split_indices


class GlmImageVectorQuantizer(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.num_embeddings, config.embed_dim)

    def encode(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.astype(mx.float32)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])

        flat = flat / (mx.linalg.norm(flat, axis=-1, keepdims=True) + 1e-6)
        embedding = self.embedding.weight.astype(mx.float32)
        embedding = embedding / (
            mx.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-6
        )

        distances = (
            mx.sum(flat * flat, axis=1, keepdims=True)
            + mx.sum(embedding * embedding, axis=1)
            - 2 * flat @ embedding.T
        )
        return mx.argmin(distances, axis=1).astype(mx.int32)


class GlmImageVQModel(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()
        self.quant_conv = nn.Conv2d(
            config.latent_channels,
            config.embed_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=True,
        )
        self.post_quant_conv = nn.Conv2d(
            config.embed_dim,
            config.latent_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=True,
        )
        self.quantize = GlmImageVectorQuantizer(config)

    def encode(self, hidden_states: mx.array) -> mx.array:
        conv_hidden_states = self.quant_conv(hidden_states)
        return self.quantize.encode(conv_hidden_states)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "quant_conv.weight" in key or "post_quant_conv.weight" in key:
                if value.ndim == 4:
                    sanitized[key] = value.transpose(0, 2, 3, 1)
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        self.vqmodel = GlmImageVQModel(config.vq_config)

    @property
    def layers(self):
        return self.language_model.model.layers

    def get_image_features(
        self,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
    ):
        hidden_states = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2
        ).tolist()
        split_indices = _compute_split_indices(split_sizes)
        return mx.split(hidden_states, split_indices, axis=0)

    def get_image_tokens(
        self,
        image_features: mx.array,
        image_grid_thw: mx.array,
    ) -> mx.array:
        split_sizes = image_grid_thw.prod(-1).tolist()
        split_indices = _compute_split_indices(split_sizes)
        features_per_image = mx.split(image_features, split_indices, axis=0)

        all_tokens = []
        hidden_size = image_features.shape[-1]
        for i, features in enumerate(features_per_image):
            grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
            features = features.reshape(
                int(grid_t), int(grid_h), int(grid_w), hidden_size
            )
            image_tokens = self.vqmodel.encode(features)
            all_tokens.append(image_tokens)

        if not all_tokens:
            return mx.array([], dtype=mx.int32)
        return mx.concatenate(all_tokens, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        images_per_sample = kwargs.get("images_per_sample", None)
        attention_mask = kwargs.get("mask", None)

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        if image_grid_thw is None:
            raise ValueError(
                "`image_grid_thw` is required when pixel_values are provided"
            )

        input_ids_np = np.array(input_ids)
        if (
            attention_mask is not None
            and attention_mask.shape[-1] == input_ids.shape[-1]
        ):
            attention_mask_np = np.array(attention_mask)
        else:
            attention_mask_np = np.ones_like(input_ids_np)

        source_image_counts = []
        for batch_idx in range(input_ids_np.shape[0]):
            sample_ids = input_ids_np[batch_idx]
            sample_mask = attention_mask_np[batch_idx] == 1
            source_image_counts.append(
                int(
                    np.sum((sample_ids == self.config.image_end_token_id) & sample_mask)
                )
            )

        if images_per_sample is not None:
            images_per_sample_np = np.array(images_per_sample).astype(np.int64)
            grids_np = np.array(image_grid_thw)
            offsets = np.cumsum([0] + images_per_sample_np.tolist())
            source_grids = []
            for sample_idx, source_count in enumerate(source_image_counts):
                sample_grids = grids_np[offsets[sample_idx] : offsets[sample_idx + 1]]
                if source_count > 0:
                    source_grids.append(sample_grids[:source_count])
            source_grids = (
                np.concatenate(source_grids, axis=0)
                if source_grids
                else np.empty((0, 3), dtype=np.int64)
            )
        else:
            source_grids = np.array(image_grid_thw)

        if len(source_grids) > 0:
            source_grids_mx = mx.array(source_grids, dtype=mx.int64)
            image_features = self.get_image_features(pixel_values, source_grids_mx)
            image_features = mx.concatenate(image_features, axis=0)
            image_ids = self.get_image_tokens(image_features, source_grids_mx)

            flat_input_ids = input_ids_np.reshape(-1)
            placeholder_positions = np.where(
                flat_input_ids == self.config.image_token_id
            )[0]
            image_ids_np = np.array(image_ids).astype(flat_input_ids.dtype)

            if placeholder_positions.shape[0] != image_ids_np.shape[0]:
                raise ValueError(
                    f"Number of image placeholder tokens ({placeholder_positions.shape[0]}) does not match "
                    f"number of image tokens from VQ model ({image_ids_np.shape[0]})"
                )

            flat_input_ids[placeholder_positions] = image_ids_np
            input_ids = mx.array(
                flat_input_ids.reshape(input_ids_np.shape), dtype=input_ids.dtype
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            images_per_sample=images_per_sample,
            attention_mask=attention_mask,
        )
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=mask,
            **kwargs,
        )

        return self.language_model(
            input_ids,
            input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        def transform_key(key):
            if key.startswith("model."):
                key = key.replace("model.", "", 1)

            if key.startswith("language_model"):
                key = key.replace("language_model", "language_model.model", 1)

            if key.startswith("visual"):
                key = key.replace("visual", "vision_model", 1)

            if key.startswith("lm_head") and not key.startswith("language_model"):
                key = key.replace("lm_head", "language_model.lm_head", 1)

            return key

        sanitized = {}
        for key, value in weights.items():
            if "model.vqmodel" in key:
                key = key.replace("model.vqmodel", "vqmodel", 1)
            new_key = transform_key(key)
            sanitized[new_key] = value

        sanitized = self.vision_model.sanitize(sanitized)
        sanitized = self.vqmodel.sanitize(sanitized)
        return sanitized
