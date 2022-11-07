import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


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
             
          
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]))


def serialize_example(img, label):
  """
  Creates a tf.train.Example message ready to be written to a file.
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
    train_writer = tf.io.TFRecordWriter(os.path.join('Data', 'train.tfrecord'))
    test_writer = tf.io.TFRecordWriter(os.path.join('Data', 'test.tfrecord'))
    val_writer = tf.io.TFRecordWriter(os.path.join('Data', 'val.tfrecord'))
    
    for folder in sorted(os.listdir(path)):
        print(folder)
        if folder == 'Training' or folder == 'Test':
            for i, fruit in enumerate(sorted(os.listdir(os.path.join(path, folder)))):
                print(i, fruit)
                
                if folder == 'Test':
                    val_idx = np.random.randint(0, len(os.listdir(os.path.join(path, folder, fruit))), 
                                                size=len(os.listdir(os.path.join(path, folder, fruit))) // 2)
                
                for j, file in enumerate(sorted(os.listdir(os.path.join(path, folder, fruit)))):
                    img = read_image(os.path.join(path, folder, fruit, file))
                    x = img.flatten() / 255
                    x = x.reshape(1, -1)
                    y = tf.one_hot(i, no_classes)
                    
                    if folder == 'Training':
                        train_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), np.asanyarray(y, dtype=np.float32)))
                    elif j in val_idx:
                        val_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), np.asanyarray(y, dtype=np.float32)))
                    else:
                        test_writer.write(serialize_example(np.asanyarray(x, dtype=np.float32), np.asanyarray(y, dtype=np.float32)))


def _parse_function(example_proto):
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    
    element = tf.io.parse_single_example(example_proto, feature_description)
    decoded_img = tf.io.parse_tensor(element['img'], 'float32')
    decoded_label = tf.io.parse_tensor(element['label'], 'float32')

    
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return decoded_img, decoded_label


def load_data_tf(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    parsed_dataset = raw_dataset.map(_parse_function)
    
    return parsed_dataset


def get_model_tf(input_shape, no_classes):
    model_input = Input(shape=input_shape)
    h1 = Dense(1024, activation='relu')(model_input)
    h2 = Dense(512, activation='relu')(h1)
    h3 = Dense(512, activation='relu')(h2)
    h4 = Dense(256, activation='relu')(h3)
    h5 = Dense(256, activation='relu')(h4)
    h6 = Dense(256, activation='relu')(h5)
    output = Dense(no_classes, activation='softmax')(h6)
    
    model = Model(inputs=model_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


