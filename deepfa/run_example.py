from utils import read_data_by_path
from deepfa import DeepFA
import numpy as np
import sys 


if len(sys.argv) != 4:
    sys.exit('\tusage run_example.py <perc of sup samples> <opf_confidence_threshold> <iterations>')

perc_sup_samples = float(sys.argv[1])
conf_threshold = float(sys.argv[2])
iterations = int(sys.argv[3])

# batch, epochs, n_classes
feat_learn_params = [32, 15, 10]

# reading data
_, x, y = read_data_by_path('../data/')

# randomly choosing the supervised samples
idx_sup_samples = np.random.randint(0, high=x.shape[0], size=(int(x.shape[0]*perc_sup_samples),))

# DeepFA params: data, labels, indexes of supervised samples, 
# feature learning parameters, confidence threshold for OPFSemi, 
# looping iterations
y_labeled = DeepFA(x, y, idx_sup_samples, feat_learn_params, conf_threshold, iterations)

