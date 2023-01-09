import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf

from skimage import io
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def read_image(path):
    """
    Read image in format HWC for RGB or HW for Grayscale from path.

    Parameters
    ----------
    path : string
        Path to image.-

    Returns
    -------
    2D or 3D numpy.ndarray
        Image in format HWC for RGB or HW for Grayscale.
    """
    image = io.imread(path)
    return image


def show_image(img, title, min=0, max=255, gray=False):
    """
    Plot image.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be plotted, in HW or HWC format.
    title : string
        Title of the plot.
    min : int, optional
        Minimum value for the plot, by default 0.
    max : int, optional
        Maximum value for the plot, by default 255.
    gray : bool, optional
        True if the image is grayscale, False otherwise, by default False.
    """
    plt. figure()
    if gray:
        plt.imshow(img, cmap='gray', vmin=min, vmax=max)
        plt.colorbar()
    else:
        plt.imshow(img, vmin=min, vmax=max)

    plt.title(title)
    plt.show()


def parse_dataset(path):
    """
    Parse dataset and print which images are not in the correct format.

    Parameters
    ----------
    path : string
        Path to dataset.
    """
    splits = [os.path.join(path, 'Training'), os.path.join(path, 'Test')]

    for folder in splits:
        for i, fruit in enumerate(sorted(os.listdir(folder))):
            print(i, fruit)
            for file in os.listdir(os.path.join(folder, fruit)):
                img = read_image(os.path.join(folder, fruit, file))
                if not np.array_equal(img.shape, (100, 100, 3)) or img.max() <= 1 or img.dtype != np.uint8:
                    print(folder, fruit, file)


def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.

    Parameters
    ----------
    value : numpy.ndarray
        Value to be converted.

    Returns
    -------
    tensorflow.train.Feature
        Feature to be written in the tfrecord file.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]))


def serialize_example(img, label):
    """
    Create a tf.train.Example message ready to be written to a file.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be serialized.
    label : tensorflow.Tensor
        Label to be serialized.

    Returns
    -------
    tensorflow.train.Example
        Example to be written in the tfrecord file.
    """
    
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'img': _bytes_feature(img),
        'label': _bytes_feature(label),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
            
                    
def get_data_tf(path, seed, no_classes):
    """
    Write tfrecord files.

    Parameters
    ----------
    path : string
        Path to dataset.
    seed : int
        Seed for the random generator.
    no_classes : int
        Number of classes.
    """
    train_writer = tf.io.TFRecordWriter(os.path.join('..', 'Data', 'train.tfrecord'))
    test_writer = tf.io.TFRecordWriter(os.path.join('..', 'Data', 'test.tfrecord'))
    val_writer = tf.io.TFRecordWriter(os.path.join('..', 'Data', 'val.tfrecord'))
    
    np.random.seed(seed)
    
    for folder in sorted(os.listdir(path)):
        if folder == 'Training' or folder == 'Test':
            print(folder)
            for i, fruit in enumerate(sorted(os.listdir(os.path.join(path, folder)))):
                print(i, fruit)
                
                if folder == 'Test':
                    val_idx = np.random.choice(len(os.listdir(os.path.join(path, 'Test', fruit))), 
                                  len(os.listdir(os.path.join(path, 'Test', fruit))) // 2, replace=False)
                
                for j, file in enumerate(sorted(os.listdir(os.path.join(path, folder, fruit)))):
                    img = read_image(os.path.join(path, folder, fruit, file))
                    x = img.flatten() / 255                    
                    y = tf.one_hot(i, no_classes)
                    
                    if folder == 'Training':
                        train_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), y))
                    elif j in val_idx:
                        val_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), y))
                    else:
                        test_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), y))



def get_data_tf_pca(path, seed, no_classes, no_components):
    """
    Write tfrecord files with PCA.

    Parameters
    ----------
    path : string
        Path to dataset.
    seed : int
        Seed for the random generator.
    no_classes : int
        Number of classes.
    no_components : int
        Number of components for PCA.
    """
    train_writer = tf.io.TFRecordWriter(os.path.join('..', 'Data', 'train_pca' + str(int(np.sqrt(no_components // 3))) + '.tfrecord'))
    test_writer = tf.io.TFRecordWriter(os.path.join('..', 'Data', 'test_pca' + str(int(np.sqrt(no_components // 3))) + '.tfrecord'))
    val_writer = tf.io.TFRecordWriter(os.path.join('..', 'Data', 'val_pca' + str(int(np.sqrt(no_components // 3))) + '.tfrecord'))
    
    np.random.seed(seed)
    pca = IncrementalPCA(n_components=no_components)

    print('Fitting PCA')
    x_count = 0
    train_files = 67692
    x_train_small = []
    for i, fruit in enumerate(sorted(os.listdir(os.path.join(path, 'Training')))):
        print(i, fruit) 
        for j, file in enumerate(sorted(os.listdir(os.path.join(path, 'Training', fruit)))):
            img = read_image(os.path.join(path, 'Training', fruit, file))
            x = img.flatten() / 255                    
            x_train_small.append(x)
            if x_count != 0 and x_count % no_components == 0:
                pca.partial_fit(x_train_small)
                x_train_small = []
            x_count += 1
        if train_files - x_count < no_components:
            break    
    
    del x_train_small
    
    print('Train PCA')
    for i, fruit in enumerate(sorted(os.listdir(os.path.join(path, 'Training')))):
        print(i, fruit) 
        for j, file in enumerate(sorted(os.listdir(os.path.join(path, 'Training', fruit)))):
            img = read_image(os.path.join(path, 'Training', fruit, file))
            x = img.flatten() / 255     
            x = x.reshape(1, -1)
            x = np.squeeze(pca.transform(x))
            y = tf.one_hot(i, no_classes)
            
            train_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), y))
            
    print('Train PCA done')
    
    for i, fruit in enumerate(sorted(os.listdir(os.path.join(path, 'Test')))):
        print(i, fruit) 
        val_idx = np.random.choice(len(os.listdir(os.path.join(path, 'Test', fruit))), 
                                  len(os.listdir(os.path.join(path, 'Test', fruit))) // 2, replace=False)
        for j, file in enumerate(sorted(os.listdir(os.path.join(path, 'Test', fruit)))):
            img = read_image(os.path.join(path, 'Test', fruit, file))
            x = img.flatten() / 255
            x = x.reshape(1, -1)
            x = np.squeeze(pca.transform(x))                    
            y = tf.one_hot(i, no_classes)
            
            if j in val_idx:
                val_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), y))
            else:
                test_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), y))
    
    print('Test PCA done')
                

def _parse_function(example_proto):
    """
    Parse a single example.

    Parameters
    ----------
    example_proto : tensorflow.string
        Example proto.

    Returns
    -------
    decoded_img : tensorflow.Tensor
        Decoded image.
    decoded_label : tensorflow.Tensor
        Decoded label.
    """
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    
    element = tf.io.parse_single_example(example_proto, feature_description)
    decoded_img = tf.io.parse_tensor(element['img'], 'float32')
    decoded_label = tf.io.parse_tensor(element['label'], 'float32')

    return decoded_img, decoded_label


def load_data_tf(path):
    """
    Load data from tfrecord files.

    Parameters
    ----------
    path : string
        Path to tfrecord file.

    Returns
    -------
    tensorflow.data.Dataset
        Dataset with images and labels.
    """
    raw_dataset = tf.data.TFRecordDataset(path)
    parsed_dataset = raw_dataset.map(_parse_function)
    
    return parsed_dataset


def get_model_tf(input_shape, no_classes):
    """
    Get simple MLP model with 6 hidden layers.

    Parameters
    ----------
    input_shape : list or tuple of int
        Shape of input.
    no_classes : int
        Number of classes, used as number of neurons in the output layer of the Neural Network.

    Returns
    -------
    tensorflow.keras.Model
        Model of MLP with 6 hidden layers.
    """
    model_input = Input(shape=input_shape)
    h1 = Dense(1024, activation='relu')(model_input)
    h2 = Dense(512, activation='relu')(h1)
    h3 = Dense(512, activation='relu')(h2)
    h4 = Dense(512, activation='relu')(h3)
    h5 = Dense(256, activation='relu')(h4)
    h6 = Dense(256, activation='relu')(h5)
    output = Dense(no_classes, activation='softmax')(h6)
    
    model = Model(inputs=model_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model


def fixup_shape(images, labels):
    """
    Fixup shape of images and labels.

    Parameters
    ----------
    images : tensorflow.Tensor
        Image to be fed into the model.
    labels : tensorflow.Tensor
        Label used for the output of the model.

    Returns
    -------
    images: tensorflow.Tensor
        Image with a fixed shape.
    labels: tensorflow.Tensor
        Label with a fixed shape.
    """
    images.set_shape([None, 10 * 10 * 3])
    labels.set_shape([None, 131])
    return images, labels


def get_confusion_matrix(y_true, y_pred, fruit_list, plot_cm=True, scale=10, print_acc=False):
    """
    Get confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    fruit_list : list or tuple of shape (n_samples,) 
        List of labels.
    plot_cm : bool, optional
        True if the confusion matrix is to be plotted, False otherwise, by default True.
    scale : int, optional
        Scale of the plotted confusion matrix for better reading, by default 10.
    print_acc : bool, optional
        True to print the per-class accuracy for every class, False otherwise, by default False.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    if plot_cm:
        cm_cut = cm[:scale, :scale]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_cut, display_labels=fruit_list[:scale])
        disp.plot()
        plt.show()
    
    if print_acc:
        for i, fruit in enumerate(cm):
            print(f'{fruit_list[i]}: {np.round(100 * fruit[i],2 )}% accuracy' )
    
    
def get_roc_auc(y_true, y_pred, no_classes):
    """
    Get ROC AUC.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes), should be in categorical format
        True labels.
    y_pred : array-like of shape (n_samples, n_classes), should be in categorical format
        Predicted labels.
    no_classes : int
        Number of classes.
    """
    fpr, tpr, roc_auc = dict(), dict(), dict()
    colors = sorted(mcolors.CSS4_COLORS.keys())
    
    plt.figure()
    for i in range(no_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i])
    plt.plot([0, 1], [0, 1], color=colors[no_classes], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
        
    
    


