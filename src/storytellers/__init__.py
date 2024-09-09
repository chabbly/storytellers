import storytellers.viewer as viewer
import storytellers.sdxl_turbo as sdxl
import storytellers.camera as camera
# from diffusers.utils import load_image


def main() -> int:
    try:
        while True:
            webcam_frame = camera.get_image()
            image = sdxl.predict(
                webcam_frame, "a scene from a Philip K. Dick novel", 0.7, 2
            )
            viewer.show_image(image)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close_viewer()
        camera.cleanup()
    return 0
