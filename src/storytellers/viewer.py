import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()


def show_image(image):
    ax.clear()
    ax.imshow(image)
    ax.axis("off")  # Hide axes
    plt.draw()
    plt.pause(0.001)  # Small pause to update the plot


def close_viewer():
    plt.close()
