from transformers import pipeline
import torch

# NOTE: this doesn't work - still ends up on CPU
device = torch.device("mps")
pipe = pipeline(
    "image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device
)

print(pipe.device)


def apply(image):
    # pillow_mask = pipe(image_path, return_mask=True)  # outputs a pillow mask
    no_bg_image = pipe(image)  # applies mask on input and returns a pillow image
    return no_bg_image
