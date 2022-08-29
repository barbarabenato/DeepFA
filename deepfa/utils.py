from tensorflow.keras.preprocessing.image import load_img
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def read_data_by_path(filename):
    """ 
    Read data from data/

    :param filename: the name of directory with data 
    return: name of image files, images samples, and its labels
    """
    txt_file = os.listdir(filename)
    imgfile = []
    label = []

    for line in range(len(txt_file)):
        imgfile.append(filename + txt_file[line].split()[0])
        label.append(int((txt_file[line].split()[0]).split('_')[0]))

    label = np.array(label)-1 

    imgs = np.array([np.array(load_img(im))
        for im in imgfile],'f')

    return imgfile, imgs, label

def save_projection(filename, data, labels, samples):
    """ 
    It saves the tSNE projection of labeled data

    :param filename: filename of the projection to be saved
    :param data: data features
    :param labels: data labels
    :param samples: index of samples to be considered as supervised
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    unsup_samples = np.arange(0, data.shape[0])
    unsup_samples = np.setdiff1d(unsup_samples, samples)
    ax.scatter(data[unsup_samples,0],data[unsup_samples,1],c=labels[unsup_samples],s=20, cmap='tab10', alpha=0.5, edgecolors='none')
    ax.scatter(data[samples,0],data[samples,1],c=labels[samples],s=20, cmap='tab10', edgecolors='red')

    plt.savefig(filename)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

