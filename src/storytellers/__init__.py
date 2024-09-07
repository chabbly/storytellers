import storytellers.sdxl_turbo as sdxl
import storytellers.camera as camera
from diffusers.utils import load_image


def main() -> int:
    webcam_frame = camera.get_image()
    webcam_frame.save("in.jpg")
    webcam_frame = load_image("in.jpg")
    # print("First 16 RGB values:", [tuple(map(int, rgb)) for rgb in first_16_rgb])
    image = sdxl.image_to_image(
        "a pear and an apple sitting happliy together", webcam_frame
    )

    image.save("out.jpg")
    camera.cleanup()
    return 0
