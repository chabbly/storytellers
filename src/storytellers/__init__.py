import storytellers.viewer as viewer
import storytellers.sdxl_turbo as sdxl
import storytellers.camera as camera
import storytellers.bg_remover as bg_remover
# from diffusers.utils import load_image


def main() -> int:
    try:
        while True:
            webcam_frame = camera.get_image()
            image = bg_remover.apply(webcam_frame)
            # image = sdxl.predict(webcam_frame, "cubism meets pointillism", 0.7, 2)
            viewer.show_image(image)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close_viewer()
        camera.cleanup()
    return 0
