import matplotlib.pyplot as plt
import skimage.io as io
import os
import numpy as np

dataset = os.path.join(os.path.join('..', 'fruits-360', 'Training'))
random_fruits = np.random.choice(os.listdir(dataset), 9)
random_images = [np.random.choice(os.listdir(os.path.join(dataset, fruit))) for fruit in random_fruits]
print(random_images)
print(random_fruits)

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i, image in enumerate(random_images):
    ax[i//3, i%3].imshow(io.imread(os.path.join(dataset, random_fruits[i], image)))
    ax[i//3, i%3].set_title(random_fruits[i])
    ax[i//3, i%3].set_xticks([])
    ax[i//3, i%3].set_yticks([])
plt.show()