import warnings
warnings.simplefilter('ignore')
import numpy as np
import imp
import time
import matplotlib.pyplot as plt
import os
import tempfile
import h5py
import keras
import keras.backend
from keras.models import load_model
from keras import activations
import matplotlib
matplotlib.use('Agg')
import innvestigate
import innvestigate.applications
import innvestigate.applications.mnist
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
from Losses import *

def rms(x):
        n = x.shape[0]
        sqsum = ( np.square(x).sum(axis=0) )
        return np.sqrt(sqsum/n)


def plot_god(weight,cpf_labels):
        cpf_w = np.sum(weight,axis=1)
        cpf_mean = np.mean(cpf_w,axis=0)
        cpf_w_rms = rms(cpf_w)
        cpf_rms_args = np.argsort(cpf_w_rms)
        cpf_w_rms_sort = cpf_w_rms[cpf_rms_args[:]]
        cpf_arguments = np.argsort(cpf_mean)
        cpf_std = np.std(cpf_w,axis=0)
        cpf_std_args = np.argsort(cpf_std)
        cpf_mean_sorted = cpf_mean[cpf_arguments[:]]
        cpf_std_sorted = cpf_std[cpf_arguments[:]]
        cpf_mean_std_sorted = cpf_mean[cpf_std_args[:]]
        cpf_std_std_sorted = cpf_std[cpf_std_args[:]]
        #cpf_covariance = np.cov(cpf_w.transpose())
        cpf_kong = []
        cpf_std_kong = []
        cpf_rms_kong = []
        for n in range(0, len(cpf_arguments)):
                cpf_kong.append(cpf_labels[cpf_arguments[n]])
                cpf_std_kong.append(cpf_labels[cpf_std_args[n]])
                cpf_rms_kong.append(cpf_labels[cpf_rms_args[n]])
                
        return cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_kong, cpf_std_kong, cpf_rms_kong, cpf_w_rms_sort

def plot_god2(weight,cpf_labels):
        cpf_w = weight
        cpf_mean = np.mean(cpf_w,axis=0)
        cpf_w_rms = rms(cpf_w)
        cpf_rms_args = np.argsort(cpf_w_rms)
        cpf_w_rms_sort = cpf_w_rms[cpf_rms_args[:]]
        cpf_arguments = np.argsort(cpf_mean)
        cpf_std = np.std(cpf_w,axis=0)
        cpf_std_args = np.argsort(cpf_std)
        cpf_mean_sorted = cpf_mean[cpf_arguments[:]]
        cpf_std_sorted = cpf_std[cpf_arguments[:]]
        cpf_mean_std_sorted = cpf_mean[cpf_std_args[:]]
        cpf_std_std_sorted = cpf_std[cpf_std_args[:]]
        #cpf_covariance = np.cov(cpf_w.transpose())
        cpf_kong = []
        cpf_std_kong = []
        cpf_rms_kong = []
        for n in range(0, len(cpf_arguments)):
                cpf_kong.append(cpf_labels[cpf_arguments[n]])
                cpf_std_kong.append(cpf_labels[cpf_std_args[n]])
                cpf_rms_kong.append(cpf_labels[cpf_rms_args[n]])
                
        return cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_kong, cpf_std_kong, cpf_rms_kong, cpf_w_rms_sort

        
def apply_modifications(model, custom_objects=None):
        """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
        # The strategy is to save the modified model and load it back. This is done because setting the activation
        # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
        # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
        # multiple inbound and outbound nodes are allowed with the Graph API.
        model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        try:
            model.save(model_path)
            return load_model(model_path, custom_objects=custom_objects)
        finally:
            os.remove(model_path)
                                            

import gc
gc.enable()
eutils = imp.load_source("utils", "utils.py")
imgnetutils = imp.load_source("utils_imagenet", "utils_imagenet.py")

x_global = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_features_0.npy')
x_cpf = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_features_1.npy')
x_npf = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_features_2.npy')
x_sv = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_features_3.npy')
x_reg = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_features_4.npy')
means = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_meansandnorms.npy')
w0 = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_weights_0.npy')
w1 = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_weights_1.npy')
y0 = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_truth_0.npy')
y1 = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_truth_1.npy') 
blab=load_model('DataStuff/DeepFlavour_noRNN_nosoftmax.h5', custom_objects=global_loss_list)
blab.summary()
global_w = np.array([])
cpf_w = np.array([])
npf_w = np.array([])
sv_w = np.array([])
a = 0
b = 299
events = 20000
firsttime = True
while b < events:
        print(a, " to ", b)
        while a < b:
                x_global_temp = x_global[a:b,:]
                x_cpf_temp = x_cpf[a:b,:,:]
                x_npf_temp = x_npf[a:b,:,:]
                x_sv_temp = x_sv[a:b,:,:]
                y0_temp = y0[a:b,:]
                inputs = [x_global_temp,x_cpf_temp,x_npf_temp,x_sv_temp]
                #analyzer = innvestigate.create_analyzer("lrp.sequential_preset_b", blab)
                analyzer = innvestigate.create_analyzer("gradient", blab)
                analysis = analyzer.analyze(inputs)
                if firsttime:
                        global_w = analysis[0]
                        cpf_w = analysis[1]
                        npf_w = analysis[2]
                        sv_w = analysis[3]
                        firsttime = False
                else:
                        global_w = np.concatenate([global_w,analysis[0]])
                        cpf_w = np.concatenate([cpf_w,analysis[1]])
                        npf_w = np.concatenate([npf_w,analysis[2]])
                        sv_w = np.concatenate([sv_w,analysis[3]])
                a = a+300
        b = b+300

#np.save("lrpweights/global_lrp_weights.npy",global_w)
#np.save("lrpweights/cpf_lrp_weights.npy",cpf_w)
#np.save("lrpweights/npf_lrp_weights.npy",npf_w)
#np.save("lrpweights/sv_lrp_weights.npy",sv_w)
np.save("lrpweights/global_gradient_weights.npy",global_w)
np.save("lrpweights/cpf_gradient_weights.npy",cpf_w)
np.save("lrpweights/npf_gradient_weights.npy",npf_w)
np.save("lrpweights/sv_gradient_weights.npy",sv_w)
