from model import *
from data import *
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import classification_report, confusion_matrix

datasetPath = "C:/Users/DiogoNunes/Documents/Tesselo/data/tesselo-training-tiles"
resultsPath = "C:/Users/DiogoNunes/Documents/Tesselo/data/results/predict"
modelsPath = "C:/Users/DiogoNunes/Documents/Tesselo/unet/models"
cleanPathsFile = "C:/Users/DiogoNunes/Documents/Tesselo/data/clean_paths.txt"
dataStatsFile = "C:/Users/DiogoNunes/Documents/Tesselo/data/data_stats.txt"
modelFilePath = os.path.join(modelsPath, "unet_COS.hdf5")

trainSize = -1 # -1 for all
testSize = -1 # -1 for all
input_size = (256, 256)
target_size = (256, 256)
target_names = ['Artificializados', 'Agricolas', 'Floresta', 'Humidas', 'Agua']
batch_size = 8
epochs = 30

ignoreNODATA_flag = False

class BatchLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('acc'))

for ni in range(2):

    use_max, do_batch_normalization, use_transpose_convolution,keepNODATA, save_header = parameters(ni)
    if ignoreNODATA_flag:
        class_labels = np.arange(5).tolist()
    else:
        class_labels = [0, 1, 2, 3, 4, 9]
    n_class = len(class_labels)

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

    trainGene = trainGeneratorCOS(batch_size, datasetPath, trainSet, dataStats, train_augmentation_args,
                                  input_size = input_size, target_size = target_size, num_classes = n_class, use_max = use_max)
    valGene = trainGeneratorCOS(batch_size, datasetPath, testSet, dataStats, val_augmentation_args,
                                input_size = input_size, target_size = target_size, num_classes = n_class, use_max = use_max)
    Ntrain = len(trainSet)
    steps_per_epoch = np.ceil(Ntrain/batch_size)
    NVal = len(testSet)
    validation_steps = np.ceil(NVal/batch_size)

    model = unet_v3(input_size = (256, 256, len(channels)), num_class = n_class, do_batch_normalization = do_batch_normalization,
                    use_transpose_convolution = use_transpose_convolution)

    model_checkpoint = ModelCheckpoint(modelFilePath, monitor = 'loss', verbose = 1, save_best_only = True)
    r.seed(1)
    history = model.fit_generator(trainGene, steps_per_epoch = steps_per_epoch, epochs = epochs,
                                  callbacks = [model_checkpoint, batch_history], validation_data = valGene,
                                  validation_steps = validation_steps)

    print('\nRunning Test Set...')
    testGene = testGeneratorCOS(datasetPath, testSet, dataStats, input_size = input_size, use_max = use_max)
    NTest = len(testSet)
    results = model.predict_generator(testGene, NTest, verbose = 0)

    print('Saving results...')
    y_gt = np.zeros(((len(testSet),) + target_size))
    y_predict = np.zeros(((len(testSet),) + target_size))
    y_gt, y_predict = saveResultCOS(datasetPath, testSet, results, resultsPath, target_size)
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

    with open('results.txt', 'a') as res_file:
        i = 1
        res_file.write('\n_______________________________________________________________')
        res_file.write(save_header)
        res_file.write('\nTraining results:')
        res_file.write('\n\t    Loss Accuracy Val_Loss  Val_Acc\n')
        for line in results_matrix:
            res_file.write('Epoch %2d' % i)
            np.savetxt(res_file, line, fmt = '%8.2f', delimiter = ' ')
            i += 1
        i = 0

        res_file.write('\nConfusion Matrix:\n\t\t  Artifici Agricola Floresta  Humidas     Agua\n')
        for line in conf_mat:
            res_file.write('%17s|' % target_names[i])
            np.savetxt(res_file, line, fmt = '%8d', delimiter = ' ')
            i += 1
        res_file.write('\nClassification report:\n')
    res_file.close()
    res_file = open('results.txt', 'ab')
    pickle.dump(class_rep, res_file)
    res_file.close()
