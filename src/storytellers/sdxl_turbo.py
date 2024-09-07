import torch
from diffusers import AutoPipelineForImage2Image

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
# hardcoded for my laptop for now, will make configurable later
pipe.to("mps")


def image_to_image(prompt, input_image):
    output_image = pipe(
        prompt,
        image=input_image,
        # if num steps < 4, it errors out for some reason
        num_inference_steps=4,
        guidance_scale=0.0,
        strength=0.3,
    ).images[0]

    return output_image
