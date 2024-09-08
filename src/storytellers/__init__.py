import storytellers.sdxl_turbo as sdxl
import storytellers.camera as camera
from diffusers.utils import load_image


def main() -> int:
    # webcam_frame = camera.get_image()
    # webcam_frame.save("in.jpg")

    webcam_frame = load_image("racoon.jpg")
    image = sdxl.predict(webcam_frame, "an albino polar bear", 0.7, 2)

    image.save("out.jpg")
    camera.cleanup()
    return 0
