"""Stable Diffusion v1.4 generation extracted from the experiment notebook.

Model imports and downloads happen only when a pipeline is requested. The
pipeline configuration, latent shape, and per-image seeds match the notebook.
"""

from .protocol import MODEL_NAME


def load_pipeline(device="cuda"):
    """Load a fresh, unmodified fp16 Stable Diffusion v1.4 pipeline."""
    import torch
    from diffusers import StableDiffusionPipeline

    return StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)


def generate_img(
    pipe,
    prompt,
    num_img=12,
    negative_prompt=None,
    seed=None,
    num_inference_steps=50,
):
    """Generate images using the notebook's explicit seeded latent tensors.

    Image ``i`` uses ``seed + i * 100``. The generator is created on the
    UNet device, and the (1, channels, 64, 64) latent uses the UNet dtype.
    The helper retains its original 50-step default; experiment protocols
    explicitly pass their validated 15-step setting.
    """
    import torch

    with torch.no_grad():
        res = []

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        for i in range(num_img):
            if seed is not None:
                generator = torch.Generator(device=pipe.unet.device).manual_seed(
                    seed + i * 100
                )
                latents = torch.randn(
                    (1, pipe.unet.config.in_channels, 64, 64),
                    generator=generator,
                    device=pipe.unet.device,
                    dtype=pipe.unet.dtype,
                )
                image = pipe(
                    prompt=prompt,
                    latents=latents,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                )[0]
            else:
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                )[0]
            res.append(image)

        return [images[0] for images in res]
