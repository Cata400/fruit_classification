import datetime

from utils_tf import *

if __name__ == '__main__': 
    dataset_path = os.path.join('..', 'fruits-360')
    fruit_list = sorted(os.listdir(os.path.join(dataset_path, 'Training')))
    no_classes = len(fruit_list) 
    
    tensorflow = True
    train = False
    extract_data = True
    pca = True
    no_components = 66 ** 2 * 3
    
    seed = 42
    batch_size = 512
    shuffle_buffer_size = 16 * batch_size
    save_model_name = 'tf_pca66_3.h5'
    callbacks = [
                TensorBoard(log_dir='../Logs/log_' + save_model_name.split('.')[0] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                ModelCheckpoint(os.path.join('..', 'Models', save_model_name), monitor='val_loss', verbose=1, save_best_only=True),
    ]
    epochs = 150

    if tensorflow:
        if train:
        ### Get data
            if extract_data:
                if pca:
                    get_data_tf_pca(dataset_path, seed, no_classes, no_components)
                else:
                    get_data_tf(dataset_path, seed, no_classes)
                exit()  
                
            train_dataset = load_data_tf(os.path.join('..', 'Data', 'train.tfrecord'))
            val_dataset = load_data_tf(os.path.join('..', 'Data', 'val.tfrecord'))
            
            train_dataset = train_dataset.cache()
            val_dataset = val_dataset.cache()
        
            ### Batch and shuffle the datasets
            tf.random.set_seed(seed)
            train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
            val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
            
            train_dataset = train_dataset.map(fixup_shape)
            val_dataset = val_dataset.map(fixup_shape)


            
            ### Initialize the model and train
            model = get_model_tf((100 * 100 * 3,), no_classes)
            model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, verbose=1)
            
    print("Gata proiectul, 10!!")
