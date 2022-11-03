from utils_numpy import *


if __name__ == '__main__':
    numpy = True
    dataset_path = os.path.join('..', 'fruits-360')

    if numpy:
        path = os.path.join('..', 'fruits-360', 'Training',
                            'Apple Granny Smith', '0_100.jpg')

        img = read_image(path)
        print(img.shape, img.min(), img.max())
        show_image(img, 'Măr')

        parse_dataset(os.path.join('..', 'fruits-360'))

    print("Gata proiectul, 10!!")
