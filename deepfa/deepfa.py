from utils import save_projection
from feature_learning import vgg16_learning
from label_propag import OPFSemi

from sklearn.manifold import TSNE
from sklearn.metrics import cohen_kappa_score, accuracy_score

import numpy as np
import os

def DeepFA(x, y, samples, feat_learn_params, conf_threshold, iterations):
    """ 
    Executes iterations of DeepFA 

    :param x: data features
    :param y: data labels
    :param samples: index of samples to be considered as supervised
    :param feat_learn_parames: list containing batch size (b), number of epochs (e), 
    and number of classes (c) [b, e, c] 
    :param conf_threshold: threshold value to be used for OPFSemi confidence [0,1]. 
    Samples with threshold up to the confidence will be selected to retrain the feature learning in the next iteration. To select all samples, conf_threshold=0.0
    :param iterations: number of DeepFA iterations
    return: pseudolabels
    """
    batch, epochs, n_classes = feat_learn_params
    y_labeled = y

    if not os.path.exists("output/"):
        os.makedirs("output/")
    file1 = open("output/results.txt", "w")

    for i in range(iterations):
        # training deep feature learning with supervised samples in the first iteration
        print("iter[%d] feature learning" % (i))
        feats_nd = vgg16_learning(x, y_labeled, samples, batch=batch, epochs=epochs, file_name='output/learning_curve_iter'+str(i)+'.png', n_classes=n_classes)

        # projecting with tsne algorithm
        print("iter[%d] computing 2d projection" % (i))
        tsne = TSNE(n_components=2, method='exact', verbose=0)
        feats_2d = tsne.fit_transform(feats_nd)

        # propagating labels with OPFSemi
        print("iter[%d] propagating labels" % (i))
        y_labeled, weights = OPFSemi(feats_2d, y, samples, conf_threshold)

        # calculating metrics
        acc = accuracy_score(y, y_labeled)
        prop_acc = cohen_kappa_score(y, y_labeled)
        file1.write("iter: %d\t\t acc: %f\t kappa: %f \n" % (i,acc,prop_acc))

        save_projection('output/tsne_iter'+str(i)+'.png', feats_2d, y_labeled, samples)

        # selecting samples for the next iteration of the looping
        if conf_threshold == 0.0: # all samples
            samples = np.random.randint(0, high=x.shape[0], size=(int(x.shape[0]),))
        else: # up the certainty value
            samples = np.argwhere(weights>=conf_threshold)
            samples = samples.reshape(samples.shape[0],)
        
    file1.close()
    return y_labeled

