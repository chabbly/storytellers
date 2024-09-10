import storytellers.viewer as viewer
import storytellers.gen_ai as gen_ai
import storytellers.image as image_utils
import storytellers.assets as assets
# from diffusers.utils import load_image

IMAGE_SIZE = 512


def main() -> int:
    frame_index = 1
    try:
        while True:
            webcam_frame = image_utils.resize_crop(
                image_utils.get_camera_frame(), IMAGE_SIZE
            )
            video_frame = assets.read_image("nggyu", frame_index)
            if video_frame is None:
                frame_index = 1
                video_frame = assets.read_image("nggyu", frame_index)
            else:
                frame_index += 1
            image = image_utils.chroma_key(video_frame, webcam_frame)
            # image = ai.predict(image, "cubism meets pointillism", IMAGE_SIZE, 0.3, 2)
            viewer.show_image(image)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close_viewer()
        image_utils.cleanup()
    return 0
