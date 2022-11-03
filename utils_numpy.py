import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os



def read_image(path):
    image = io.imread(path)

    return image

def show_image(img, title, min=0, max=255, gray=False):
    plt. figure()
    if gray:
        plt.imshow(img, cmap='gray', vmin=min, vmax=max)
        plt.colorbar()
    else:
        plt.imshow(img, vmin=min, vmax=max)

    plt.title(title)
    plt.show()

def parse_dataset(path):
    splits = [os.path.join(path, 'Training'), os.path.join(path, 'Test')]

    for folder in splits:
        for i, fruit in enumerate(sorted(os.listdir(folder))):
            print(i, fruit)
            for file in os.listdir(os.path.join(folder, fruit)):
                img = read_image(os.path.join(folder, fruit, file))
                if not np.array_equal(img.shape, (100, 100, 3)) or img.max() <= 1 or img.dtype != np.uint8:
                    print(folder, fruit, file)