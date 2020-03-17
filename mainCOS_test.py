from model import *
from data import *
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import classification_report, confusion_matrix

main_path = "C:\\Users\DiogoSilipeSilvaNunes\PycharmProjects\SatForest" # Modificar conforme necessário

datasetPath = os.path.join(main_path, "data\\tesselo-training-tiles")
resultsPath = os.path.join(main_path, "data\\results\predict")
modelsPath = os.path.join(main_path, "umodels")
cleanPathsFile = os.path.join(main_path, "data\clean_paths.txt")
dataStatsFile = os.path.join(main_path, "data\data_stats.txt")
modelFilePath = os.path.join(modelsPath, "unet_COS.hdf5")
resultsFilePath = os.path.join(main_path, "data\\results\\results.txt")

trainSize = -1 # -1 for all
testSize = -1 # -1 for all
target_names = ['Artificializados', 'Agricolas', 'Floresta', 'Humidas', 'Agua']

use_unet = 5
batch_size_train = 52
batch_size_test = 52
epochs = 30

if use_unet == 3:
    input_size = (256, 256)
    target_size = (256, 256)
else:
    input_size = (252, 252)
    target_size = (68, 68) # (164, 164)
ignoreNODATA_flag = True
keepNODATA = False

class BatchLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('acc'))

trainSet, testSet, dataStats = prepare_dataset(datasetPath = datasetPath, ignoreNODATAtiles = ignoreNODATA_flag,
                                               keepNODATA = keepNODATA, cleanTilesFile = cleanPathsFile,
                                               dataStatsFile = dataStatsFile)

train_augmentation_args = dict(width_shift_range = [0, 1, 2, 3],
                               height_shift_range = [0, 1, 2, 3],
                               rotation_range = [0, 90, 180, 270],
                               horizontal_flip = True,
                               vertical_flip = True)

val_augmentation_args = dict()
batch_history = BatchLossHistory()
if trainSize > 0:
    trainSet = trainSet[0:trainSize]
if testSize > 0:
    testSet = testSet[0:testSize]

for ni in range(1):
    keras.backend.clear_session()

    use_max, do_batch_normalization, use_transpose_convolution, net_channels, drop, save_header = parameters(ni)

    if not ignoreNODATA_flag and keepNODATA:
        class_labels = [0, 1, 2, 3, 4, 9]
    else:
        class_labels = np.arange(10).tolist() # <--------------
    n_class = len(class_labels)

    trainGene = trainGeneratorCOS(batch_size_train, datasetPath, trainSet, dataStats, train_augmentation_args,
                                  input_size = input_size, target_size = target_size, num_classes = n_class,
                                  use_max = use_max, ignoreNODATA_flag = ignoreNODATA_flag, keepNODATA = keepNODATA,
                                  use_unet = use_unet)

    valGene = trainGeneratorCOS(batch_size_test, datasetPath, testSet, dataStats, val_augmentation_args,
                                input_size = input_size, target_size = target_size, num_classes = n_class,
                                use_max = use_max, ignoreNODATA_flag = ignoreNODATA_flag, keepNODATA = keepNODATA,
                                use_unet = use_unet)

    Ntrain = len(trainSet)
    steps_per_epoch = np.ceil(Ntrain/batch_size_train)
    NVal = len(testSet)
    validation_steps = np.ceil(NVal/batch_size_test)

    if use_unet == 3:
        model = unet_v3B(input_size = (input_size[0], input_size[1], len(channels)), num_class = n_class,
                        do_batch_normalization = do_batch_normalization,
                        use_transpose_convolution = use_transpose_convolution, net_channels = net_channels,
                        dropout = drop)
    else:
        model = unet_v4(input_size = (input_size[0], input_size[1], len(channels)), num_class = n_class,
                        do_batch_normalization = do_batch_normalization,
                        use_transpose_convolution = use_transpose_convolution, net_channels = net_channels,
                        dropout = drop)
    model_checkpoint = ModelCheckpoint(modelFilePath, monitor = 'loss', verbose = 1, save_best_only = True)
    r.seed(1)
    history = model.fit_generator(trainGene, steps_per_epoch = steps_per_epoch, epochs = epochs,
                                  callbacks = [model_checkpoint, batch_history], validation_data = valGene,
                                  validation_steps = validation_steps)

    print('\nRunning Test Set...')
    testGene = testGeneratorCOS(datasetPath, testSet, dataStats, input_size = input_size, use_max = use_max)
    NTest = len(testSet)
    results = model.predict_generator(testGene, NTest, verbose = 1)

    print('Saving results...')
    y_gt = np.zeros(((len(testSet),) + target_size))
    y_predict = np.zeros(((len(testSet),) + target_size))
    y_gt, y_predict = saveResultCOS(datasetPath, testSet, results, resultsPath, target_size, keepNODATA = keepNODATA)
    print('Results saved')
    print('Calculating metrics...')
    y_gt = y_gt.reshape(-1)             #Flatten the matrices
    y_predict = y_predict.reshape(-1)
    conf_mat = confusion_matrix(y_gt, y_predict, labels = class_labels)
    class_rep = classification_report(y_gt, y_predict, labels = class_labels)
    print('\nConfusion Matrix:')
    print(conf_mat)
    print('\nClassification report:')
    print(class_rep)

    conf_mat = np.matrix(conf_mat)
    results_matrix = np.zeros((epochs, 1), dtype = np.float)
    results_matrix = np.c_[results_matrix, np.matrix(history.history['loss']).T]
    results_matrix = np.c_[results_matrix, np.matrix(history.history['acc']).T]
    results_matrix = np.c_[results_matrix, np.matrix(history.history['val_loss']).T]
    results_matrix = np.c_[results_matrix, np.matrix(history.history['val_acc']).T]
    results_matrix = np.delete(results_matrix, (0), axis = 1)

    with open(resultsFilePath, 'a') as res_file:
        res_file.write('\n_______________________________________________________________')
        res_file.write(save_header)
        res_file.write('\nTraining results:')
        res_file.write('\n\t    Loss Accuracy Val_Loss  Val_Acc\n')
        for i, line in enumerate(results_matrix):
            res_file.write('Epoch %2d' % (i + 1))
            np.savetxt(res_file, line, fmt = '%8.4f', delimiter = ' ')

        #res_file.write('\nConfusion Matrix:\n\t\t  Artifici Agricola Floresta  Humidas     Agua\n')
        res_file.write('\nConfusion Matrix:\n\t\t\t\t\t Artifici Agricult SAFazinh SeFSoAze SAFoutro FLeucali FLpinhei    Matos FLoutras Agua+Hum\n')
        #ind = 0
        for ind, line in enumerate(conf_mat):
            '''
            res_file.write('%17s|' % target_names[ind])
            np.savetxt(res_file, line, fmt = '%8d', delimiter = ' ')
            '''
            # Semi-final agregation
            res_file.write('%40s|' % tesselo_class_names[ind])
            np.savetxt(res_file, line, fmt = '%8d', delimiter = ' ')

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
