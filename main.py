import datetime

from utils_tf import *
from utils_torch import *

if __name__ == '__main__': 
    dataset_path = os.path.join('..', 'fruits-360')
    fruit_list = sorted(os.listdir(os.path.join(dataset_path, 'Training')))
    no_classes = len(fruit_list) 
    
    tensorflow = False
    pytorch = True
    
    assert tensorflow != pytorch, 'Choose only one framework: tensorflow or torch!'
    
    train = True
    extract_data = True
    pca = True
    no_components = 10 ** 2 * 3
    
    seed = 42
    batch_size = 512
    shuffle_buffer_size = 16 * batch_size
    save_model_name = 'tf_pca10_2.h5'
    callbacks = [
                TensorBoard(log_dir='../Logs/log_' + save_model_name.split('.')[0] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                ModelCheckpoint(os.path.join('..', 'Models', save_model_name), monitor='val_loss', verbose=1, save_best_only=True),
    ]
    epochs = 150
    
    load_model_name = 'tf_nopca_2.h5'

    if tensorflow:
        if pca:
            train_tfrecord = 'train_pca' + str(int(np.sqrt(no_components // 3))) + '.tfrecord'
            val_tfrecord = 'val_pca' + str(int(np.sqrt(no_components // 3))) + '.tfrecord'
            test_tfrecord = 'test_pca' + str(int(np.sqrt(no_components // 3))) + '.tfrecord'
        else:
            train_tfrecord = 'train.tfrecord'
            val_tfrecord = 'val.tfrecord'
            test_tfrecord = 'test.tfrecord'
        if train:
        ### Get data
            if extract_data:
                if pca:
                    get_data_tf_pca(dataset_path, seed, no_classes, no_components)
                else:
                    get_data_tf(dataset_path, seed, no_classes)
                exit()
                
            ### Load data
            train_dataset = load_data_tf(os.path.join('..', 'Data', train_tfrecord))
            val_dataset = load_data_tf(os.path.join('..', 'Data', val_tfrecord))
            
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
            
        else:
            ### Load data
            test_dataset = load_data_tf(os.path.join('..', 'Data', test_tfrecord))
            
            ### Get true labels
            y_true = []
            for example in test_dataset.take(-1):
                y_true.append(example[1].numpy())
            y_true = np.asanyarray(y_true)
            y_true_labels = np.argmax(y_true, axis=1)
            
            ### Batch and shuffle the datasets
            test_dataset = test_dataset.batch(batch_size)
            
            test_dataset = test_dataset.map(fixup_shape)
            
            ### Load model and predict
            model = tf.keras.models.load_model(os.path.join('..', 'Models', load_model_name))
            y_pred = model.predict(test_dataset, batch_size=batch_size, verbose=0)
            y_pred_labels = np.argmax(y_pred, axis=1)
            loss, acc = model.evaluate(test_dataset, verbose=0)
            
            print("Test accuracy: {:5.2f}%".format(100 * acc))
            
            ### Get confusion matrix
            get_confusion_matrix(y_true_labels, y_pred_labels, fruit_list, print_acc=True)
            
            ### Get ROC AUC
            get_roc_auc(y_true, y_pred, no_classes)

    elif pytorch:
        if train:
            print(f"Torch CUDA is available: {torch.cuda.is_available()}")
            
            ### Get data
            train_dataset = FruitDataset(dir=os.path.join(dataset_path, 'Training'), target_transform=OneHotEncoding(fruit_list))
            test_dataset = FruitDataset(dir=os.path.join(dataset_path, 'Test'), target_transform=OneHotEncoding(fruit_list)) #TODO: allow validation data
            
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            
            
    print("Gata proiectul, 10!!")
