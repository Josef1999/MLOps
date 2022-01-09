import numpy as np
import torch

from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os




sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),'deploy'))

from PytorchModel import PytorchModel

def test_model(*args, **kwargs):
    print("dataset to load")
    dataset_path = "test_dataset.npz"
    data = np.load(dataset_path)
    labels = data["kdd"][:,-1]
    features = data["kdd"][:,:-1]
    print("data loaded")
    dagmm = PytorchModel.DaGMM(4)
    model_name = "../deploy/model.pth"
    dagmm.load_state_dict(torch.load(model_name, map_location=torch.device('cpu') ))
    dagmm.eval()

    enc, dec, z, gamma = dagmm(Variable((torch.tensor(features).to(torch.float32))))

    test_energy = []

    sample_energy, cov_diag = dagmm.compute_energy(z, size_average=False)
    test_energy.append(sample_energy.data.cpu().numpy())
    test_energy = np.concatenate(test_energy,axis=0)
    thresh = np.percentile(test_energy, 100 - 20)

    pred = (test_energy > thresh).astype(int)
    gt = labels.astype(int)
    acc = accuracy_score(gt,pred)
    print("Accuracy: ", acc)
    assert acc > 0.90
