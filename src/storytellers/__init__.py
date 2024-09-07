import storytellers.sdxl_turbo as sdxl
import storytellers.camera as camera


def main() -> int:
    print("Hello from storytellers!")
    image = sdxl.image_to_image(None, "a pear and an apple sitting happliy together")
    print(image)

    image = camera.get_image()
    image.save("out.jpg")

    camera.cleanup()
    return 0
