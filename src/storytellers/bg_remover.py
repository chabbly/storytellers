from PIL import Image
import numpy as np


def combine(bottom_image, top_image):
    # Convert images to numpy arrays
    bottom_array = np.array(bottom_image)
    top_array = np.array(top_image)

    # Define the white-ish color range
    lower_white = np.array([180, 180, 180])
    upper_white = np.array([255, 255, 255])

    # Create a mask for white-ish pixels
    mask = np.all((top_array >= lower_white) & (top_array <= upper_white), axis=-1)

    # Use the mask to combine the images
    result = np.where(mask[:, :, np.newaxis], bottom_array, top_array)

    # Convert back to PIL Image
    return Image.fromarray(result)
