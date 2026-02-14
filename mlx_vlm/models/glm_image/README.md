# GLM-Image

GLM-Image support in `mlx-vlm` adds first-class loading for the `glm_image` model family, including:

- Custom model wiring for `model_type="glm_image"`
- Processor support for target image grid tokens
- Source-image placeholder expansion and image tokenization flow

## Quick start

```python
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "zai-org/GLM-Image/vision_language_encoder"
model, processor = load(model_path, trust_remote_code=True)

prompt = "A cinematic photo of a red vintage car parked in the rain"
formatted = apply_chat_template(processor, model.config, prompt)
```

## Notes

- The processor appends target image grids to prompts and returns `image_grid_thw` + `images_per_sample`.
- Source images are converted to image tokens before language decoding.
- `glm_image` uses dedicated image start/end token IDs and vision token vocab semantics.
