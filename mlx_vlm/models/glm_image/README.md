# GLM-Image autoregressive encoder

This package loads GLM-Image's vision-language encoder, turns source images into VQ codebook tokens, and predicts image-token IDs with the autoregressive language model. Target image grids determine the spatial positions used during decoding.

GLM-Image combines this encoder with a diffusion decoder. Rendering an image requires that separate decoder; this package provides the autoregressive component. See the [Transformers GLM-Image model documentation](https://huggingface.co/docs/transformers/model_doc/glm_image) and [Diffusers GLM-Image pipeline](https://huggingface.co/docs/diffusers/api/pipelines/glm_image).

A local snapshot can keep encoder weights and config in `vision_language_encoder/` and tokenizer/image metadata in `processor/`. Loading resolves those folders. Conversion flattens the relevant metadata into the converted encoder directory, preserving the new weight index and avoiding copies of sibling diffusion weights.

```python
from mlx_vlm import load
from mlx_vlm.generate import generate_step
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

# A local zai-org/GLM-Image snapshot, or its converted encoder directory.
model, processor = load("/path/to/GLM-Image")
prompt = apply_chat_template(processor, model.config, "A red car in the rain")
inputs = prepare_inputs(processor, prompts=prompt, target_h=512, target_w=512)

tokens = generate_step(
    inputs["input_ids"],
    model,
    inputs.get("pixel_values"),
    inputs.get("attention_mask"),
    image_grid_thw=inputs["image_grid_thw"],
    images_per_sample=inputs["images_per_sample"],
    max_tokens=16,
    temperature=0,
)
image_token_ids = [int(token) for token, _ in tokens]
```

For image-conditioned prompts, pass `images=[...]` to `prepare_inputs` and supply the same source-image count to `apply_chat_template`. Preprocessing uses NumPy and Pillow; it does not require PyTorch or torchvision. Each batch must contain the same number of source images per prompt.

Synthetic tests cover vision/VQ features, prompt grids, prefill and incremental spatial decoding, processor serialization, nested checkpoint loading, and conversion reload. Real checkpoint output quality and native Metal execution still require runtime verification.
