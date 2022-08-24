import pyift.pyift as ift
import sys
import numpy as np


def OPFSemi(x, y, samples, conf=0.0):
    # creating opfdataset with provided features
    Z = ift.CreateDataSetFromNumPy(x, np.array(y+1, dtype="int32")) # opf dataset considers labels as [1,n]
    # defining status as training (4) for all samples
    stat = np.full((x.shape[0],), fill_value=(4), dtype="uint32")
    # defining status as supervised (68) for supervised samples
    stat[samples] = 68
    Z.SetStatus(np.array(stat, dtype="int32"))
    Z.SetTrueLabels(np.array(y+1, dtype="int32"))
    Z.SetNTrainSamples(x.shape[0])

    # creating graph and obtaining certainty values
    graph = ift.SemiSupTrain(Z)
    # calculating confidence values if confidence > 0
    if conf > 0.0:
        stat[stat == 4] = 2
        Z.SetStatus(np.array(stat, dtype="int32"))
        ift.ClassifyWithCertaintyValues(graph, Z)
    
    labels = Z.GetLabels()-1
    weights = Z.GetWeights()

    return labels, weights