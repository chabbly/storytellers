import cv2
from PIL import Image

# One-off initialization
camera = cv2.VideoCapture(0)


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

    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        return None

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)

    return pil_image


def cleanup():
    """
    Releases the camera resource.
    Should be called when the application is closing.
    """
    global camera
    if camera.isOpened():
        camera.release()
