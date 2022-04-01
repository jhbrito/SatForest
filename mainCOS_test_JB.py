import os.path

from model_unet_COS import input_valid_sizes, output_valid_sizes, unetL3, unetL4, unetL5, unetL6, unet
from data import *
import tensorflow as tf
from COS_train_options import *
from sklearn.metrics import classification_report, confusion_matrix

print("Tensorflow {}".format(tf.__version__))
if tf.test.is_gpu_available():
    print("GPU available: {}".format(tf.test.gpu_device_name()))
else:
    print("GPU not available")

# modelFileName = 'unetCOSV1_ClassesCOSN1B_NClasses4_Level5_Featuremaps64_Padsame_BNNone_TCFalse_Dropout0.5_Batchsize8_Epochs200_Datetime20211029-145644.hdf5'
modelFileName = 'unetCOSV1_ClassesCOSN3_NClasses32_Level5_Featuremaps64_Padsame_BNNone_TCFalse_Dropout0.5_Batchsize8_Epochs200_Datetime20211031-133707.hdf5'
# modelFileName = 'unetCOSV1_ClassesCOSN5_NClasses48_Level5_Featuremaps64_Padsame_BNNone_TCFalse_Dropout0.5_Batchsize8_Epochs200_Datetime20211104-175340.hdf5'
clean_paths_file = "./data/clean_paths_V1.txt"
data_stats_file = "./data/data_stats_V1.txt"
models_path = "./models"
modelFilePath = os.path.join(models_path, modelFileName)

if not os.path.exists(results_path):
    os.makedirs(results_path)

history_path = os.path.join(results_path, "history")
if not os.path.exists(history_path):
    os.makedirs(history_path)

predict_path = os.path.join(results_path, "predict")
if not os.path.exists(predict_path):
    os.makedirs(predict_path)

confusion_matrix_path = os.path.join(results_path, "confusion_matrix")
if not os.path.exists(confusion_matrix_path):
    os.makedirs(confusion_matrix_path)

classification_report_path = os.path.join(results_path, "classification_report")
if not os.path.exists(classification_report_path):
    os.makedirs(classification_report_path)

trainSet, testSet, dataStats = prepare_dataset_V1(datasetPath=dataset_path, cleanTilesFile=clean_paths_file, dataStatsFile=data_stats_file)
if trainSize > 0:
    trainSet = trainSet[0:trainSize]
if testSize > 0:
    testSet = testSet[0:testSize]


if padding == 'valid':
    input_size = (input_valid_sizes[level_i], input_valid_sizes[level_i])
    target_size = (output_valid_sizes[level_i], output_valid_sizes[level_i])
else:
    input_size = (256, 256)
    target_size = (256, 256)
batch_size_train = batch_size
batch_size_test = batch_size_train

n_classes = len(class_labels)
filename_options = os.path.splitext(modelFileName)[:-1][0]

results_predict_path = os.path.join(predict_path, "predictV1" + filename_options)
confusion_matrix_file_path = os.path.join(confusion_matrix_path, "confusion_matrixV1" + filename_options + ".pickle")
classification_report_file_path = os.path.join(classification_report_path, "classification_reportV1" + filename_options + ".pickle")
resultsFilePath = os.path.join(results_path, "results" + filename_options + ".txt")


print('\nRunning Test Set...')
testGene = testGeneratorCOS(dataset_path, testSet, dataStats, input_size=input_size, use_max=use_max)
NTest = len(testSet)

# load the model
if unet_level == 3:
    model = unetL3(pretrained_weights=modelFilePath,
                   input_size=(input_size[0], input_size[1], len(channels)),
                   num_class=n_classes,
                   net_channels=net_channels,
                   padding=padding,
                   batch_normalization=batch_normalization,
                   use_transpose_convolution=use_transpose_convolution,
                   dropout=dropout)
elif unet_level == 4:
    model = unetL4(pretrained_weights=modelFilePath,
                   input_size=(input_size[0], input_size[1], len(channels)),
                   num_class=n_classes,
                   net_channels=net_channels,
                   padding=padding,
                   batch_normalization=batch_normalization,
                   use_transpose_convolution=use_transpose_convolution,
                   dropout=dropout)
elif unet_level == 5:
    model = unetL5(pretrained_weights=modelFilePath,
                   input_size=(input_size[0], input_size[1], len(channels)),
                   num_class=n_classes,
                   net_channels=net_channels,
                   padding=padding,
                   batch_normalization=batch_normalization,
                   use_transpose_convolution=use_transpose_convolution,
                   dropout=dropout)
elif unet_level == 6:
    model = unetL6(pretrained_weights=modelFilePath,
                   input_size=(input_size[0], input_size[1], len(channels)),
                   num_class=n_classes,
                   net_channels=net_channels,
                   batch_normalization=batch_normalization,
                   use_transpose_convolution=use_transpose_convolution,
                   dropout=dropout)
else:
    # default unet model from zhixuhao
    model = unet(pretrained_weights=modelFilePath,
                 input_size=(input_size[0], input_size[1], len(channels)),
                 num_class=n_classes)

results = model.predict_generator(testGene, NTest, verbose=1)

print('Saving results...')
y_gt = np.zeros(((len(testSet),) + target_size))
y_predict = np.zeros(((len(testSet),) + target_size))
y_gt, y_predict = saveResultCOS(dataset_path, testSet, results, results_predict_path, target_size, export_COS_files=export_COS_files)
print('Results saved')

print('Calculating metrics...')
y_gt = y_gt.reshape(-1)  # Flatten the matrices
y_predict = y_predict.reshape(-1)

print('Calculating Confusion Matrix...')
conf_mat = confusion_matrix(y_gt, y_predict, labels=class_labels)
print('Saving Confusion Matrix...')
with open(confusion_matrix_file_path, "wb") as cm:  # Pickling
    pickle.dump(conf_mat, cm)
print('Confusion Matrix:')
print(conf_mat)

print('Calculating Classification Report...')
class_rep = classification_report(y_gt, y_predict, labels=class_labels)
print('Saving Classification Report...')
with open(classification_report_file_path, "wb") as cr:  # Pickling
    pickle.dump(class_rep, cr)
print('\nClassification report:')
print(class_rep)

with open(resultsFilePath, 'a') as res_file:
    res_file.write('_______________________________________________________________')
    res_file.write(save_header)
    res_file.write('\nConfusion Matrix:\n')
    for ind, line in enumerate(conf_mat):
        res_file.write('%40s |' % class_aggregation_names[ind])
        np.savetxt(res_file, line, fmt='%8d', delimiter=' ')
    res_file.write('\nClassification report:\n')
    pickle.dump(class_rep, res_file)
