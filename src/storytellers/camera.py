import cv2
from PIL import Image

# One-off initialization
camera = cv2.VideoCapture(0)


def resize_crop(image, width=512):
    image = image.convert("RGB")
    w, h = image.size
    image = image.resize((width, int(width * (h / w))), Image.BICUBIC)
    return image


def get_image():
    """
    Captures and returns the current webcam image as a PIL Image.

    Returns:
    - PIL.Image: The captured image.
    - None: If the capture fails.
    """
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return None

    # Set camera parameters for brightness (not working yet)
    # camera.set(cv2.CAP_PROP_BRIGHTNESS, 255)  # Adjust brightness (0-255)
    # this is a *gross* way to set a hex value
    # camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # camera.set(cv2.CAP_PROP_EXPOSURE, -7)  # Adjust exposure (-7 to -1 for manual mode)

    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        return None

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)

    return resize_crop(pil_image)


def cleanup():
    """
    Releases the camera resource.
    Should be called when the application is closing.
    """
    global camera
    if camera.isOpened():
        camera.release()
