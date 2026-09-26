"""Notebook-equivalent CLIP cosine scoring and single-sample evaluation."""

import gc

from . import generation
from .projection import build_projection, project_text_conder_attn
from .protocol import (
    CLIP_MODEL_NAME,
    LAMBDA_VALUE,
    LAYER_ID,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
)


def clip_car_scores(model, processor, device, imgs, prompt):
    """Return mean normalized CLIP cosine similarity per image, on CPU.

    The historical helper name is retained; ``prompt`` may describe any
    concept. This is the notebook's unscaled cosine metric, with identical
    handling of transformers feature outputs exposing ``pooler_output``.
    """
    import torch

    with torch.inference_mode():
        text_inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
        ).to(device)
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_inputs = processor(images=imgs, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        return similarity.mean(dim=1).cpu()


class CLIPScorer:
    """Own the notebook's CLIP model and processor for repeated evaluations.

    Importing this module does not load or download models. Constructing the
    scorer loads ``openai/clip-vit-large-patch14`` once, in evaluation mode.
    """

    def __init__(self, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def score(self, images, prompt):
        """Return the mean cosine similarity as a Python scalar."""
        return clip_car_scores(
            self.model, self.processor, self.device, images, prompt
        ).mean().item()


def evaluate_single(
    U_concept,
    rank,
    prompt,
    clip_prompt,
    seed,
    scorer,
    device="cuda",
):
    """Generate and score one sample using a fresh pipeline for every call.

    ``rank=None`` evaluates the original model for preservation comparisons.
    Otherwise, the cached concept basis is projected using the validated
    lambda and text-encoder layer. Pipeline cleanup also runs on failure.
    """
    import torch

    pipe = generation.load_pipeline(device=device)
    try:
        if rank is not None:
            projection = build_projection(U_concept, rank, lambda_value=LAMBDA_VALUE)
            project_text_conder_attn(pipe, projection, layer_id=LAYER_ID)

        images = generation.generate_img(
            pipe,
            prompt,
            num_img=1,
            negative_prompt=NEGATIVE_PROMPT,
            seed=seed,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )
        return scorer.score(images, clip_prompt)
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
