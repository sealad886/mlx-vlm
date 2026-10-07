import math

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ImagesKwargs, ProcessorMixin

from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import _resize_video_frames, _to_numpy_image


def _round_to_multiple(value: int, factor: int) -> int:
    return max((int(value) // factor) * factor, factor)


class GlmImageImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int


class GlmImageImageProcessor(ImageProcessingMixin):
    """NumPy/PIL preprocessing for GLM-Image's half-resolution source grids."""

    valid_kwargs = GlmImageImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size=16,
        temporal_patch_size=1,
        merge_size=1,
        size=None,
        min_pixels=None,
        max_pixels=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = size or {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
        if min_pixels is not None:
            self.size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            self.size["longest_edge"] = max_pixels
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std or [0.26862954, 0.26130258, 0.27577711]

    def __call__(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]
        patches, grids = [], []
        ps, ms = self.patch_size, self.merge_size
        factor = ps * ms
        shortest_edge = round(math.sqrt(self.size["shortest_edge"]))
        longest_edge = round(math.sqrt(self.size["longest_edge"]))
        for image in images:
            image = _to_numpy_image(image)
            channels, height, width = image.shape
            if (
                min(height, width) < factor
                or max(height, width) / min(height, width) > 4
            ):
                raise ValueError(
                    "GLM-Image sources must span a patch and have aspect ratio <= 4"
                )
            scale = max(1.0, shortest_edge / min(height, width))
            if max(height, width) * scale > longest_edge:
                scale = longest_edge / max(height, width)
            resized_h = max(factor, round((height // 2) * scale / factor) * factor)
            resized_w = max(factor, round((width // 2) * scale / factor) * factor)
            if max(resized_h, resized_w) > longest_edge:
                beta = max(resized_h, resized_w) / longest_edge
                resized_h = max(factor, math.floor(resized_h / beta / factor) * factor)
                resized_w = max(factor, math.floor(resized_w / beta / factor) * factor)
            values = _resize_video_frames(image[None], resized_h, resized_w)[0].astype(
                np.float32
            )
            if self.do_rescale:
                values *= self.rescale_factor
            if self.do_normalize:
                mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
                std = np.array(self.image_std, dtype=np.float32)[:, None, None]
                values = (values - mean) / std
            h, w = resized_h // ps, resized_w // ps
            values = values.reshape(channels, h // ms, ms, ps, w // ms, ms, ps)
            values = values.transpose(1, 4, 2, 5, 0, 3, 6)
            values = np.repeat(
                values[..., None, :, :], self.temporal_patch_size, axis=-3
            )
            patches.append(
                values.reshape(h * w, channels * self.temporal_patch_size * ps * ps)
            )
            grids.append([1, h, w])
        return {
            "pixel_values": np.concatenate(patches),
            "image_grid_thw": np.array(grids, dtype=np.int64),
        }

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)


class GlmImageProcessor(ProcessorMixin):
    requires_target_image_grid = True
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.image_token = getattr(tokenizer, "image_token", "<|image|>")
        self.grid_bos_token = getattr(tokenizer, "grid_bos_token", "<sop>")
        self.grid_eos_token = getattr(tokenizer, "grid_eos_token", "<eop>")
        self.bos_token = getattr(tokenizer, "bos_token", "")
        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.image_token)
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids")
            else 167855
        )

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _build_prompt_with_target_shape(
        self,
        prompt: str,
        target_h: int,
        target_w: int,
        is_text_to_image: bool,
    ):
        factor = 32
        target_h = _round_to_multiple(target_h, factor)
        target_w = _round_to_multiple(target_w, factor)

        token_h = target_h // factor
        token_w = target_w // factor

        ratio = token_h / max(token_w, 1)
        prev_token_h = max(int(math.sqrt(ratio) * (factor // 2)), 1)
        prev_token_w = max(int(math.sqrt(1 / ratio) * (factor // 2)), 1)

        if is_text_to_image:
            expanded_prompt = (
                f"{prompt}"
                f"{self.grid_bos_token}{token_h} {token_w}{self.grid_eos_token}"
                f"{self.grid_bos_token}{prev_token_h} {prev_token_w}{self.grid_eos_token}"
                f"{self.bos_token}"
            )
        else:
            expanded_prompt = f"{prompt}{self.grid_bos_token}{token_h} {token_w}{self.grid_eos_token}{self.bos_token}"

        return expanded_prompt, token_h, token_w, prev_token_h, prev_token_w

    @staticmethod
    def _build_target_image_grid_thw(
        token_h: int,
        token_w: int,
        prev_token_h: int,
        prev_token_w: int,
        is_text_to_image: bool = True,
    ) -> np.ndarray:
        if is_text_to_image:
            return np.array(
                [[1, token_h, token_w], [1, prev_token_h, prev_token_w]],
                dtype=np.int64,
            )
        return np.array([[1, token_h, token_w]], dtype=np.int64)

    def __call__(
        self,
        images=None,
        text: str | list[str] = None,
        target_h: int = 1152,
        target_w: int = 768,
        **kwargs,
    ) -> BatchFeature:
        padding = kwargs.pop("padding", False)
        return_token_type_ids = kwargs.pop("return_token_type_ids", False)
        return_tensors = kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", False)

        image_inputs = {}
        image_grid_thw = np.empty((0, 3), dtype=np.int64)
        if images is not None and self.image_processor is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = np.array(
                image_inputs.get("image_grid_thw", []), dtype=np.int64
            )

        is_text_to_image = images is None

        if text is None:
            if images is None:
                raise ValueError("You must provide at least one of `text` or `images`.")
            return BatchFeature(data=image_inputs, tensor_type=return_tensors)

        if not isinstance(text, list):
            text = [text]

        text = [str(t) for t in text]
        batch_size = len(text)
        images_per_sample = [sample.count(self.image_token) for sample in text]

        if (
            not is_text_to_image
            and images_per_sample
            and len(set(images_per_sample)) != 1
        ):
            raise ValueError(
                "In image-to-image mode, all samples must have the same number of source images. "
                f"Got different counts: {images_per_sample}"
            )

        if not is_text_to_image:
            required_source_images = sum(images_per_sample)
            if required_source_images > len(image_grid_thw):
                raise ValueError(
                    "Number of source image placeholders does not match available image grids"
                )

            source_grid_index = 0
            for i in range(batch_size):
                while self.image_token in text[i]:
                    grid = image_grid_thw[source_grid_index]
                    _, grid_h, grid_w = grid.tolist()
                    num_image_tokens = int(grid_h * grid_w)
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    source_grid_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        num_source_images = images_per_sample[0] if images_per_sample else 0
        all_grids = []
        for i in range(batch_size):
            text[i], token_h, token_w, prev_h, prev_w = (
                self._build_prompt_with_target_shape(
                    text[i],
                    target_h=target_h,
                    target_w=target_w,
                    is_text_to_image=is_text_to_image,
                )
            )

            if not is_text_to_image and num_source_images > 0:
                start_idx = i * num_source_images
                all_grids.append(
                    image_grid_thw[start_idx : start_idx + num_source_images]
                )

            all_grids.append(
                self._build_target_image_grid_thw(
                    token_h=token_h,
                    token_w=token_w,
                    prev_token_h=prev_h,
                    prev_token_w=prev_w,
                    is_text_to_image=is_text_to_image,
                )
            )

        combined_grid_thw = (
            np.concatenate(all_grids, axis=0)
            if all_grids
            else np.empty((0, 3), dtype=np.int64)
        )
        image_inputs["image_grid_thw"] = combined_grid_thw
        target_grid_count = 2 if is_text_to_image else 1
        image_inputs["images_per_sample"] = np.array(
            [num_source_images + target_grid_count] * batch_size,
            dtype=np.int64,
        )

        text_inputs = self.tokenizer(  # noqa: E501 #type: ignore
            text,
            padding=padding,
            return_token_type_ids=return_token_type_ids,
            **kwargs,
        )

        data = {**text_inputs, **image_inputs}

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            data["mm_token_type_ids"] = mm_token_type_ids.tolist()

        if return_tensors == "mlx":
            from ..base import to_mlx

            return BatchFeature(data=to_mlx(data))
        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        return_dict=False,
        return_tensors=None,
        target_h: int = 1152,
        target_w: int = 768,
        **kwargs,
    ):
        rendered_prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if not tokenize:
            return rendered_prompt

        images = []
        for message in conversation:
            if not isinstance(message, dict):
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"image", "input_image", "image_url"}:
                    image_ref = item.get("url") or item.get("image")
                    image_url = item.get("image_url")
                    if image_ref is None and isinstance(image_url, dict):
                        image_ref = image_url.get("url")
                    if image_ref is None and isinstance(image_url, str):
                        image_ref = image_url
                    if image_ref is not None:
                        images.append(image_ref)

        model_inputs = self(
            images=images if images else None,
            text=[rendered_prompt],
            target_h=target_h,
            target_w=target_w,
            return_tensors=return_tensors,
        )

        if return_dict:
            return model_inputs
        return model_inputs["input_ids"]

    @property
    def model_input_names(self):
        tokenizer_input_names = (
            self.tokenizer.model_input_names if self.tokenizer else []
        )
        image_processor_input_names = (
            self.image_processor.model_input_names
            if hasattr(self.image_processor, "model_input_names")
            else []
        )
        extra_names = ["image_grid_thw", "images_per_sample"]
        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names + extra_names
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        trust_remote_code = kwargs.pop("trust_remote_code", True)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        from ..base import load_chat_template

        load_chat_template(tokenizer, pretrained_model_name_or_path)
        kwargs.setdefault("chat_template", getattr(tokenizer, "chat_template", None))

        image_processor = GlmImageImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **{key: value for key, value in kwargs.items() if key != "chat_template"},
        )

        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)


__all__ = ["GlmImageProcessor", "GlmImageImageProcessor"]

install_auto_processor_patch(["glm_image", "glm_image_text"], GlmImageProcessor)
