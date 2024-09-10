import storytellers.viewer as viewer
import storytellers.sdxl_turbo as sdxl
import storytellers.camera as camera
import storytellers.bg_remover as bg_remover
import storytellers.assets as assets
# from diffusers.utils import load_image


def main() -> int:
    frame_index = 1
    try:
        while True:
            webcam_frame = sdxl.resize_crop(camera.get_image())
            video_frame = assets.read_image("nggyu", frame_index)
            if video_frame is None:
                frame_index = 1
                video_frame = assets.read_image("nggyu", frame_index)
            else:
                frame_index += 1
            image = bg_remover.combine(video_frame, webcam_frame)
            # image = sdxl.predict(image, "cubism meets pointillism", 0.3, 2)
            viewer.show_image(image)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close_viewer()
        camera.cleanup()
    return 0
