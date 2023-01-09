from utils_tf import *


image_path = os.path.join('..', 'fruits-360', 'Training', 'Apple Braeburn', '0_100.jpg')
dataset_path = os.path.join('..', 'fruits-360')
no_components = 50 ** 2 * 3

print('Fitting PCA')
pca = IncrementalPCA(n_components=no_components)
x_count = 0
train_files = 67692
x_train_small = []
for i, fruit in enumerate(sorted(os.listdir(os.path.join(dataset_path, 'Training')))):
    print(i, fruit) 
    for j, file in enumerate(sorted(os.listdir(os.path.join(dataset_path, 'Training', fruit)))):
        img = read_image(os.path.join(dataset_path, 'Training', fruit, file))
        x = img.flatten() / 255                    
        x_train_small.append(x)
        if x_count != 0 and x_count % no_components == 0:
            pca.partial_fit(x_train_small)
            x_train_small = []
        x_count += 1
    if train_files - x_count < no_components:
        break    

del x_train_small

img = read_image(image_path)
x = img.flatten() / 255     
x = x.reshape(1, -1)
x = np.squeeze(pca.transform(x))

img_pca = pca.inverse_transform(x).reshape(100, 100, 3)

print("Total information conserved: ", 100 * np.sum(pca.explained_variance_ratio_), "%")


fig, ax = plt.subplots(1, 3, figsize=(10, 10))
ax[0].imshow(img), ax[0].set_title('Original image'), ax[0].set_xticks([]), ax[0].set_yticks([])
ax[1].imshow(x.reshape(int(np.sqrt(no_components // 3)), int(np.sqrt(no_components // 3)), 3)), ax[1].set_title('PCA transformed image'), ax[1].set_xticks([]), ax[1].set_yticks([])
ax[2].imshow(img_pca), ax[2].set_title('Reconstructed image'), ax[2].set_xticks([]), ax[2].set_yticks([])
plt.show()
