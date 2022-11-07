from utils_tf import *


if __name__ == '__main__':  
    tf = True
    train = True
    dataset_path = os.path.join('..', 'fruits-360')
    extract_data = False
    fruit_list = sorted(os.listdir(os.path.join(dataset_path, 'Training')))
    no_classes = len(fruit_list)
    seed = 42
    batch_size = 1024
    shuffle_buffer_size = 16 * batch_size
    save_model_name = 'tf_1.h5'
    callbacks = [
                TensorBoard(log_dir='Logs'),
                ModelCheckpoint(os.path.join('Models', save_model_name), monitor='val_loss', verbose=1, save_best_only=True),
    ]
    epochs = 1000

    if tf:
        if train:
        ### Get data
            if extract_data:
                get_data_tf(dataset_path, seed, no_classes)   
                
            train_dataset = load_data_tf(os.path.join('Data', 'train.tfrecord'))
            val_dataset = load_data_tf(os.path.join('Data', 'val.tfrecord'))
            
            train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
            val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
            
            model = get_model_tf((100 * 100 * 3), no_classes)
            model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, verbose=1)
            
                
        
        
        

    print("Gata proiectul, 10!!")
