import storytellers.viewer as viewer
import storytellers.sdxl_turbo as sdxl
import storytellers.camera as camera
import storytellers.bg_remover as bg_remover
import storytellers.assets as assets
# from diffusers.utils import load_image


def main() -> int:
    try:
        while True:
            webcam_frame = sdxl.resize_crop(camera.get_image())
            video_frame = assets.read_image("nggyu", 4)
            image = bg_remover.combine(video_frame, webcam_frame)
            # image = sdxl.predict(webcam_frame, "cubism meets pointillism", 0.7, 2)
            viewer.show_image(image)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close_viewer()
        camera.cleanup()
    return 0
