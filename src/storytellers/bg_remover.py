from PIL import Image
import numpy as np


def apply(image):
    # Convert image to numpy array
    img_array = np.array(image)

    # Define the white-ish color range
    lower_white = np.array([180, 180, 180])
    upper_white = np.array([255, 255, 255])

    # Create a mask for white-ish pixels
    mask = np.all((img_array >= lower_white) & (img_array <= upper_white), axis=-1)

    # Create an alpha channel
    alpha = np.where(mask, 0, 255).astype(np.uint8)

    # Add alpha channel to the image
    result = np.dstack((img_array, alpha))

    # Convert back to PIL Image
    return Image.fromarray(result)
