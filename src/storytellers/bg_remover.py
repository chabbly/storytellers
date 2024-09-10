from PIL import Image
import numpy as np


def apply(image):
    # Convert image to numpy array
    img_array = np.array(image)

    # Calculate luminance
    luminance = (
        0.299 * img_array[:, :, 0]
        + 0.587 * img_array[:, :, 1]
        + 0.114 * img_array[:, :, 2]
    )

    # Create a mask based on luminance threshold (adjust as needed)
    threshold = 200  # Higher values keep darker pixels
    mask = luminance > threshold

    # Create an alpha channel
    alpha = np.where(mask, 0, 255).astype(np.uint8)

    # Add alpha channel to the image
    result = np.dstack((img_array, alpha))

    # Convert back to PIL Image
    return Image.fromarray(result)
