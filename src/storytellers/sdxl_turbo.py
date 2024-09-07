from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
# hardcoded for my laptop for now, will make configurable later
pipe.to("mps")


def image_to_image(image, prompt):
    output_image = pipe(
        prompt=prompt, num_inference_steps=1, guidance_scale=0.0
    ).images[0]
    return output_image
