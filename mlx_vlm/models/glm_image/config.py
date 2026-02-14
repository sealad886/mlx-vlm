from dataclasses import dataclass, field
from typing import Dict, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "glm_image_text"
    vocab_size: int = 168064
    hidden_size: int = 4096
    intermediate_size: int = 13696
    max_position_embeddings: int = 131072
    num_attention_heads: int = 32
    num_hidden_layers: int = 40
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rope_parameters: Dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "mrope_section": [8, 12, 12],
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000,
        }
    )
    pad_token_id: int = 167841
    use_cache: bool = True
    tie_word_embeddings: bool = False
    vision_vocab_size: int = 16512
    eos_token_id: int = 16385
    dtype: str = "bfloat16"

    @property
    def rope_theta(self) -> float:
        return self.rope_parameters.get("rope_theta", 10000)

    @property
    def partial_rotary_factor(self) -> float:
        return self.rope_parameters.get("partial_rotary_factor", 1.0)


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "glm_image_vision"
    depth: int = 40
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_heads: int = 16
    patch_size: int = 16
    image_size: int = 2048
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    spatial_merge_size: int = 1


@dataclass
class VQConfig(BaseModelConfig):
    model_type: str = "glm_image_vqmodel"
    embed_dim: int = 2048
    num_embeddings: int = 16384
    latent_channels: int = 1536
    in_channels: int = 3
    initializer_range: float = 0.02


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    vq_config: VQConfig = None
    model_type: str = "glm_image"
    image_token_id: int = 167855
    image_start_token_id: int = 16384
    image_end_token_id: int = 16385
    tie_word_embeddings: bool = False
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig(**self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig(**self.vision_config)
        if isinstance(self.vq_config, dict):
            self.vq_config = VQConfig(**self.vq_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.vq_config is None:
            self.vq_config = VQConfig()

        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id
