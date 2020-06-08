from model_unet_COS import *
from data import *
import keras
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix

dataset_path = "C:/Tesselo/data/tesselo-training-tiles"
clean_paths_file = "./data/clean_paths.txt"
data_stats_file = "./data/data_stats.txt"
models_path = "./models"

results_path = "./results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

history_path = results_path + "/history"
if not os.path.exists(history_path):
    os.makedirs(history_path)

predict_path = results_path + "/predict"
if not os.path.exists(predict_path):
    os.makedirs(predict_path)

confusion_matrix_path = results_path + "/confusion_matrix"
if not os.path.exists(confusion_matrix_path):
    os.makedirs(confusion_matrix_path)

classification_report_path = results_path + "/classification_report"
if not os.path.exists(classification_report_path):
    os.makedirs(classification_report_path)

trainSize = -1  # -1 for all
testSize = -1  # -1 for all

epochs = 2  # 50

ignoreNODATA_flag = True
keepNODATA = False

trainSet, testSet, dataStats = prepare_dataset(datasetPath=dataset_path, ignoreNODATAtiles=ignoreNODATA_flag,
                                               keepNODATA=keepNODATA, cleanTilesFile=clean_paths_file,
                                               dataStatsFile=data_stats_file)
if trainSize > 0:
    trainSet = trainSet[0:trainSize]
if testSize > 0:
    testSet = testSet[0:testSize]

train_augmentation_args = dict(width_shift_range=[0, 1, 2, 3],
                               height_shift_range=[0, 1, 2, 3],
                               rotation_range=[0, 90, 180, 270],
                               horizontal_flip=True,
                               vertical_flip=True)

val_augmentation_args = dict()


class BatchLossHistoryCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_accuracies = []
        self.batch_losses = []

    def on_train_begin(self, logs={}):
        self.batch_accuracies = []
        self.batch_losses = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('acc'))


batch_history_callback = BatchLossHistoryCallback()

if True:  # for unet_model_i in range(len(unet_models)):
    if True:  # for net_channels_i in range(len(net_channels_options)):
        # keras.backend.clear_session()

        # unet_model_i = 0
        # net_channels_i = 0
        # unet_model, net_channels, batch_size, do_batch_normalization, use_transpose_convolution, drop, use_max, save_header = experiment_parameters(
        #     unet_model_i, net_channels_i)
        # individual test
        # unet_model = 3  # [2 3 3.5 4 4.5]
        # net_channels = 64  # [32 64]
        # batch_size = 3  # [ [0, 5], [8, 3], [9, 4], [22, 8], [16, 6]]

        use_max = False
        level_i = 2  # default 2(5)
        channels_i = 1  # default 1(64)
        padding_i = 0  # default 0 (same)
        batch_normalization_i = 0  # default 0 (None)
        use_transpose_convolution_i = 0  # default 0 (False)

        unet_level, net_channels, padding, batch_normalization, use_transpose_convolution, dropout, batch_size, save_header = experiment_parameters(level_i=level_i, channels_i=channels_i, padding_i=padding_i, batch_normalization_i=batch_normalization_i, use_transpose_convolution_i=use_transpose_convolution_i)

        batch_size = 2 * 5  # 13
        if padding == 'valid':
            input_size = (input_valid_sizes[level_i], input_valid_sizes[level_i])
            target_size = (output_valid_sizes[level_i], output_valid_sizes[level_i])
        else:
            input_size = (256, 256)
            target_size = (256, 256)
        batch_size_train = batch_size
        batch_size_test = batch_size_train
        if not ignoreNODATA_flag and keepNODATA:
            class_labels = [0, 1, 2, 3, 4, 9]
        else:
            class_labels = np.arange(10).tolist()  # <--------------

        n_classes = len(class_labels)
        filename_options = "_Classes" + str(n_classes) +\
                           "_Level" + str(unet_level) + \
                           "_Featuremaps" + str(net_channels) + \
                           "_Pad" + padding + \
                           "_BN" + batch_normalization + \
                           "_TC" + str(use_transpose_convolution) + \
                           "_Dropout" + str(dropout) + \
                           "_Batchsize" + str(batch_size_train) + \
                           "_Epochs" + str(epochs) + \
                           "_Datetime" + datetime.now().strftime("%Y%m%d-%H%M%S")
        modelFilePath = os.path.join(models_path, "unetCOS" + filename_options + ".hdf5")
        logdir = "logs/scalars/l" + filename_options
        history_file_path = os.path.join(history_path, "history" + filename_options + ".pickle")
        results_predict_path = os.path.join(predict_path, "predict" + filename_options)
        confusion_matrix_file_path = os.path.join(confusion_matrix_path, "confusion_matrix" + filename_options + ".pickle")
        classification_report_file_path = os.path.join(classification_report_path, "classification_report" + filename_options + ".pickle")
        resultsFilePath = os.path.join(results_path, "results" + filename_options + ".pickle")

        trainGene = trainGeneratorCOS(batch_size_train, dataset_path, trainSet, dataStats, train_augmentation_args,
                                      input_size=input_size, target_size=target_size, num_classes=n_classes,
                                      use_max=use_max, ignoreNODATA_flag=ignoreNODATA_flag)
        valGene = trainGeneratorCOS(batch_size_test, dataset_path, testSet, dataStats, val_augmentation_args,
                                    input_size=input_size, target_size=target_size, num_classes=n_classes,
                                    use_max=use_max, ignoreNODATA_flag=ignoreNODATA_flag)

        Ntrain = len(trainSet)
        steps_per_epoch = np.ceil(Ntrain / batch_size_train)
        NVal = len(testSet)
        validation_steps = np.ceil(NVal / batch_size_test)

        if unet_level == 3:
            model = unetL3(input_size=(input_size[0], input_size[1], len(channels)),
                           num_class=n_classes,
                           net_channels=net_channels,
                           padding=padding,
                           batch_normalization=batch_normalization,
                           use_transpose_convolution=use_transpose_convolution,
                           dropout=dropout)
        elif unet_level == 4:
            model = unetL4(input_size=(input_size[0], input_size[1], len(channels)),
                           num_class=n_classes,
                           net_channels=net_channels,
                           padding=padding,
                           batch_normalization=batch_normalization,
                           use_transpose_convolution=use_transpose_convolution,
                           dropout=dropout)
        elif unet_level == 5:
            model = unetL5(input_size=(input_size[0], input_size[1], len(channels)),
                           num_class=n_classes,
                           net_channels=net_channels,
                           padding=padding,
                           batch_normalization=batch_normalization,
                           use_transpose_convolution=use_transpose_convolution,
                           dropout=dropout)
        elif unet_level == 6:
            model = unetL6(input_size=(input_size[0], input_size[1], len(channels)),
                           num_class=n_classes,
                           net_channels=net_channels,
                           batch_normalization=batch_normalization,
                           use_transpose_convolution=use_transpose_convolution,
                           dropout=dropout)
        else:
            # default unet model from zhixuhao
            model = unet(input_size=(input_size[0], input_size[1], len(channels)),
                            num_class=n_classes)

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(modelFilePath,
                                                                    monitor='val_loss',
                                                                    verbose=1,
                                                                    save_best_only=True)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        r.seed(1)
        history = model.fit_generator(trainGene,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      callbacks=[model_checkpoint_callback, batch_history_callback,
                                                 early_stopping_callback, tensorboard_callback],
                                      validation_data=valGene,
                                      validation_steps=validation_steps)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        with open(history_file_path, "wb") as hf:  # Pickling
            pickle.dump(history, hf)
        print('\nRunning Test Set...')
        testGene = testGeneratorCOS(dataset_path, testSet, dataStats, input_size=input_size, use_max=use_max)
        NTest = len(testSet)
        results = model.predict_generator(testGene, NTest, verbose=1)

        print('Saving results...')
        y_gt = np.zeros(((len(testSet),) + target_size))
        y_predict = np.zeros(((len(testSet),) + target_size))
        y_gt, y_predict = saveResultCOS(dataset_path, testSet, results, results_predict_path, target_size)
        print('Results saved')

        print('Calculating metrics...')
        y_gt = y_gt.reshape(-1)  # Flatten the matrices
        y_predict = y_predict.reshape(-1)

        print('Calculating Confusion Matrix...')
        conf_mat = confusion_matrix(y_gt, y_predict, labels=class_labels)
        print('Saving Confusion Matrix...')
        with open(confusion_matrix_file_path, "wb") as cm:  # Pickling
            pickle.dump(conf_mat, cm)
        print('\nConfusion Matrix:')
        print(conf_mat)

        print('Calculating Classification Report...')
        class_rep = classification_report(y_gt, y_predict, labels=class_labels)
        print('Saving Classification Report...')
        with open(classification_report_file_path, "wb") as cr:  # Pickling
            pickle.dump(class_rep, cr)
        print('\nClassification report:')
        print(class_rep)

        print('Saving summary...')
        conf_mat = np.array(conf_mat)
        epochs_ran = len(history.history['loss'])
        results_matrix = np.zeros((epochs_ran, 1), dtype=np.float)
        results_matrix = np.c_[results_matrix, np.array(history.history['loss']).T]
        results_matrix = np.c_[results_matrix, np.array(history.history['acc']).T]
        results_matrix = np.c_[results_matrix, np.array(history.history['val_loss']).T]
        results_matrix = np.c_[results_matrix, np.array(history.history['val_acc']).T]
        results_matrix = np.delete(results_matrix, (0), axis=1)

        with open(resultsFilePath, 'a') as res_file:
            res_file.write('\n_______________________________________________________________')
            res_file.write(save_header)
            res_file.write('\nTraining results:')
            res_file.write('\n\t    Loss Accuracy Val_Loss  Val_Acc\n')
            for i, line in enumerate(results_matrix):
                res_file.write('Epoch %2d' % (i + 1))
                np.savetxt(res_file, line, fmt='%8.4f', delimiter=' ')

            # res_file.write('\nConfusion Matrix:\n\t\t  Artifici Agricola Floresta  Humidas     Agua\n')
            res_file.write(
                '\nConfusion Matrix:\n\t\t\t\t\t Artifici Agricult SAFazinh SeFSoAze SAFoutro FLeucali FLpinhei    Matos FLoutras Agua+Hum\n')
            # ind = 0
            for ind, line in enumerate(conf_mat):
                '''
                res_file.write('%17s|' % target_names[ind])
                np.savetxt(res_file, line, fmt = '%8d', delimiter = ' ')
                '''
                # Semi-final agregation
                res_file.write('%40s|' % tesselo_class_names[ind])
                np.savetxt(res_file, line, fmt='%8d', delimiter=' ')

                '''
                # For more than 5 classes
                if ind == 0:
                    res_file.write('%49s|' % target_names[ind])
                    np.savetxt(res_file, line, fmt='%8d', delimiter=' ')
                elif ind == 30:
                    res_file.write('%49s|' % target_names[4])
                    np.savetxt(res_file, line, fmt = '%8d', delimiter = ' ')
                else:
                    res_file.write('%49s|' % class_names[class_lookup[ind + 12]])
                    np.savetxt(res_file, line, fmt='%8d', delimiter=' ')
                ind += 1
                '''
            res_file.write('\nClassification report:\n')

        with open(resultsFilePath, 'ab') as res_file:
            pickle.dump(class_rep, res_file)
