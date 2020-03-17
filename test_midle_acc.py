import os
import skimage.io as io
import numpy as np

def do_center_crop(img, new_size):
    cropy = (int)((img.shape[0] - new_size[0]) / 2)
    cropx = (int)((img.shape[1] - new_size[1]) / 2)
    img = img[cropy:img.shape[0] - cropy, cropx:img.shape[1] - cropx]
    return img

resultsPath = 'C:\\Users\DiogoNunes\Documents\Tesselo\data\\results\predict'
test_set_path = 'C:\\Users\DiogoNunes\Documents\Tesselo\data\\test_set_paths.txt'
new_size = (250, 250)
n_class = 10
total_acc = 0
lines = 0
with open(test_set_path, 'r') as test_paths:
    for line in test_paths:
        lines += 1
        filename = line
        filename = filename.replace('\n', '')
        cos_gt = io.imread(os.path.join(resultsPath, filename + '_COS_GT10.tif'), as_gray = True)
        cos_predict = io.imread(os.path.join(resultsPath, filename + '_COS_predict10.tif'), as_gray = True)
        croped_cos_gt = do_center_crop(cos_gt, new_size)
        croped_cos_predict = do_center_crop(cos_predict, new_size)
        '''
        ## This
        tot = np.zeros(cos_gt.shape, dtype = 'uint8')
        for i in range(n_class):
            tot[cos_gt == i & cos_predict == i] = 1
        '''
        ## OR this
        tot = 0
        croped_cos_gt = croped_cos_gt.reshape(-1)
        croped_cos_predict = croped_cos_predict.reshape(-1)
        for i in range(len(croped_cos_gt)):
            if croped_cos_gt[i] == croped_cos_predict[i]:
                tot += 1

        single_acc = tot / len(croped_cos_gt) # or just tot.sum() / (new_size[0] * new_size[1])
        total_acc += single_acc

total_acc /= lines
print('\nMiddle Accuracy: ' + str(total_acc))
