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

import innvestigate
import innvestigate.applications
import innvestigate.applications.mnist
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
from Losses import *


def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp((x.transpose()-x.max(axis=1)).transpose())
        return e_x / np.sum(e_x,axis=1)[:,None]

def rms(x):
        n = x.shape[0]
        sqsum = ( np.square(x).sum(axis=0) )
        return np.sqrt(sqsum/n)


def sorting(weight,cpf_labels):
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
        cpf_labels_sorted = []
        cpf_std_labels_sorted = []
        cpf_rms_labels_sorted = []
        for n in range(0, len(cpf_arguments)):
                cpf_labels_sorted.append(cpf_labels[cpf_arguments[n]])
                cpf_std_labels_sorted.append(cpf_labels[cpf_std_args[n]])
                cpf_rms_labels_sorted.append(cpf_labels[cpf_rms_args[n]])
                
        return cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_labels_sorted, cpf_std_labels_sorted, cpf_rms_labels_sorted, cpf_w_rms_sort

def sorting2(weight,cpf_labels):
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
        cpf_labels_sorted = []
        cpf_std_labels_sorted = []
        cpf_rms_labels_sorted = []
        for n in range(0, len(cpf_arguments)):
                cpf_labels_sorted.append(cpf_labels[cpf_arguments[n]])
                cpf_std_labels_sorted.append(cpf_labels[cpf_std_args[n]])
                cpf_rms_labels_sorted.append(cpf_labels[cpf_rms_args[n]])
                
        return cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_labels_sorted, cpf_std_labels_sorted, cpf_rms_labels_sorted, cpf_w_rms_sort

        
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
#x_cpf = np.load('DataStuff/NumpyConversion/Dir_features_1.npy')
#x_npf = np.load('DataStuff/NumpyConversion/Dir_features_2.npy')
#x_sv = np.load('DataStuff/NumpyConversion/Dir_features_3.npy')
#x_reg = np.load('DataStuff/NumpyConversion/Dir_features_4.npy')
#means = np.load('DataStuff/NumpyConversion/Dir_meansandnorms.npy')
#w0 = np.load('DataStuff/NumpyConversion/Dir_weights_0.npy')
#w1 = np.load('DataStuff/NumpyConversion/Dir_weights_1.npy')
y = np.load('DataStuff/DeepFlavour_numpy_ttbar/DeepFlavour_ttBar_numpy_truth_0.npy')
y = y[:19734]
jet_pt = x_global[:19734,0]
pred = np.load('prediction.npy') 
pred_soft = softmax(pred)
lrp = True
lrpw2 = False
rnn = False
smoothgrad = False
norm = True

if lrp:
        global_w = np.load("ttbarWeights/global_lrp_weights.npy")
        cpf_w = np.load("ttbarWeights/cpf_lrp_weights.npy")
        npf_w = np.load("ttbarWeights/npf_lrp_weights.npy")
        sv_w = np.load("ttbarWeights/sv_lrp_weights.npy")
elif lrpw2:
        global_w = np.load("ttbarWeights/global_lrp_w2_weights.npy")
        cpf_w = np.load("ttbarWeights/cpf_lrp_w2_weights.npy")
        npf_w = np.load("ttbarWeights/npf_lrp_w2_weights.npy")
        sv_w = np.load("ttbarWeights/sv_lrp_w2_weights.npy")	
elif smoothgrad:
        global_w = np.abs(np.load("ttbarWeights/global_smooth_gradient_weights.npy"))
        cpf_w = np.abs(np.load("ttbarWeights/cpf_smooth_gradient_weights.npy"))
        npf_w = np.abs(np.load("ttbarWeights/npf_smooth_gradient_weights.npy"))
        sv_w = np.abs(np.load("ttbarWeights/sv_smooth_gradient_weights.npy"))	

else:
        if rnn:
                global_w = np.abs(np.load("ttbarWeights_withrnn/global_gradient_weights.npy"))
                cpf_w = np.abs(np.load("ttbarWeights_withrnn/cpf_gradient_weights.npy"))
                npf_w = np.abs(np.load("ttbarWeights_withrnn/npf_gradient_weights.npy"))
                sv_w = np.abs(np.load("ttbarWeights_withrnn/sv_gradient_weights.npy"))
        else:
                global_w = np.abs(np.load("ttbarWeights/global_gradient_weights.npy"))
                cpf_w = np.abs(np.load("ttbarWeights/cpf_gradient_weights.npy"))
                npf_w = np.abs(np.load("ttbarWeights/npf_gradient_weights.npy"))
                sv_w = np.abs(np.load("ttbarWeights/sv_gradient_weights.npy"))


if norm:
        global_row_sums = np.sum(np.abs(global_w),axis=1)
        cpf_row_sums = np.sum(np.sum(np.abs(cpf_w),axis=1),axis=1)
        npf_row_sums = np.sum(np.sum(np.abs(npf_w),axis=1),axis=1)
        sv_row_sums = np.sum(np.sum(np.abs(sv_w),axis=1),axis=1)
        row_sums = sv_row_sums+npf_row_sums+cpf_row_sums+global_row_sums
        cpf_w = cpf_w / row_sums[:,None,None]
        cpf_w = cpf_w[~np.isnan(cpf_w).any(axis=1).any(axis=1)] 
        npf_w = npf_w / row_sums[:,None,None]
        npf_w = npf_w[~np.isnan(npf_w).any(axis=1).any(axis=1)]
        sv_w = sv_w / row_sums[:,None,None]
        sv_w = sv_w[~np.isnan(sv_w).any(axis=1).any(axis=1)]
        global_w = global_w / row_sums[:,None]
        global_w = global_w[~np.isnan(global_w).any(axis=1)] 


selector = True
if selector:
        #selection = ( ((y[:,4] == 1) | (y[:,5] == 1)) & ( ( pred_soft[:,0]+pred_soft[:,1]+pred_soft[:,2] ) > 0.3) )
        #selection = (jet_pt > 200.0)
        selection = ( (y[:,4] == 0) & (y[:,5] == 0)  )
        global_w = global_w[selection]
        cpf_w = cpf_w[selection]
        npf_w = npf_w[selection]
        sv_w = sv_w[selection]

cpf_labels = ['Cpfcan_BtagPf_trackEtaRel',
          'Cpfcan_BtagPf_trackPtRel',
          'Cpfcan_BtagPf_trackPPar',
          'Cpfcan_BtagPf_trackDeltaR',
          'Cpfcan_BtagPf_trackPParRatio',
          'Cpfcan_BtagPf_trackSip2dVal',
          'Cpfcan_BtagPf_trackSip2dSig',
          'Cpfcan_BtagPf_trackSip3dVal',
          'Cpfcan_BtagPf_trackSip3dSig',
          'Cpfcan_BtagPf_trackJetDistVal',
          'Cpfcan_ptrel',
          'Cpfcan_drminsv',
          'Cpfcan_VTX_ass',
          'Cpfcan_puppiw',
          'Cpfcan_chi2',
          'Cpfcan_quality'
]

cpf_even = np.linspace(1,16,16)

cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_labels_sorted, cpf_std_labels_sorted, cpf_rms_labels_sorted, cpf_w_rms_sort = sorting(cpf_w,cpf_labels)


plt.errorbar(cpf_even,cpf_mean_sorted,cpf_std_sorted)
plt.xticks(cpf_even, cpf_labels_sorted, rotation=90)
plt.show()
plt.errorbar(cpf_even,cpf_mean_std_sorted,cpf_std_std_sorted)
plt.xticks(cpf_even, cpf_std_labels_sorted, rotation=90)
plt.show()
plt.plot(cpf_even,cpf_w_rms_sort)
plt.xticks(cpf_even, cpf_rms_labels_sorted, rotation=90)
plt.show()

npf_labels = ['Npfcan_ptrel',
              'Npfcan_deltaR',
              'Npfcan_isGamma',
              'Npfcan_HadFrac',
              'Npfcan_drminsv',
              'Npfcan_puppiw'
]

npf_even = np.linspace(1,6,6)

npf_mean_sorted, npf_std_sorted, npf_mean_std_sorted, npf_std_std_sorted, npf_labels_sorted, npf_std_labels_sorted, npf_rms_labels_sorted, npf_w_rms_sort = sorting(npf_w,npf_labels)


plt.errorbar(npf_even,npf_mean_sorted,npf_std_sorted)
plt.xticks(npf_even, npf_labels_sorted, rotation=90)
plt.show()
plt.errorbar(npf_even,npf_mean_std_sorted,npf_std_std_sorted)
plt.xticks(npf_even, npf_std_labels_sorted, rotation=90)
plt.show()
plt.plot(npf_even,npf_w_rms_sort)
plt.xticks(npf_even, npf_rms_labels_sorted, rotation=90)
plt.show()

sv_labels = ['sv_pt',
             'sv_deltaR',
             'sv_mass',
             'sv_ntracks',
             'sv_chi2',
             'sv_normchi2',
             'sv_dxy',
             'sv_dxysig',
             'sv_d3d',
             'sv_d3dsig',
             'sv_costhetasvpv',
             'sv_enratio']


sv_even = np.linspace(1,12,12)
sv_mean_sorted, sv_std_sorted, sv_mean_std_sorted, sv_std_std_sorted, sv_labels_sorted, sv_std_labels_sorted, sv_rms_labels_sorted, sv_w_rms_sort = sorting(sv_w,sv_labels)

plt.errorbar(sv_even,sv_mean_sorted,sv_std_sorted)
plt.xticks(sv_even, sv_labels_sorted, rotation=90)
plt.show()
plt.errorbar(sv_even,sv_mean_std_sorted,sv_std_std_sorted)
plt.xticks(sv_even, sv_std_labels_sorted, rotation=90)
plt.show()
plt.plot(sv_even,sv_w_rms_sort)
plt.xticks(sv_even, sv_rms_labels_sorted, rotation=90)
plt.show()


global_labels = ['jet_pt', 'jet_eta','nCpfcand',
          'nNpfcand','nsv','npv',
          'TagVarCSV_trackSumJetEtRatio',
          'TagVarCSV_trackSumJetDeltaR',
          'TagVarCSV_vertexCategory',
          'TagVarCSV_trackSip2dValAboveCharm',
          'TagVarCSV_trackSip2dSigAboveCharm',
          'TagVarCSV_trackSip3dValAboveCharm',
          'TagVarCSV_trackSip3dSigAboveCharm',
          'TagVarCSV_jetNSelectedTracks',
          'TagVarCSV_jetNTracksEtaRel'
          ]

global_even = np.linspace(1,15,15)
global_mean_sorted, global_std_sorted, global_mean_std_sorted, global_std_std_sorted, global_labels_sorted, global_std_labels_sorted, global_rms_labels_sorted, global_w_rms_sort = sorting2(global_w,global_labels)

plt.errorbar(global_even,global_mean_sorted,global_std_sorted)
plt.xticks(global_even, global_labels_sorted, rotation=90)
plt.show()
plt.errorbar(global_even,global_mean_std_sorted,global_std_std_sorted)
plt.xticks(global_even, global_std_labels_sorted, rotation=90)
plt.show()
plt.plot(global_even,global_w_rms_sort)
plt.xticks(global_even, global_rms_labels_sorted, rotation=90)
plt.show()

combine_rms = np.concatenate((global_w_rms_sort,sv_w_rms_sort,npf_w_rms_sort,cpf_w_rms_sort))
combine_label = np.concatenate((global_rms_labels_sorted,sv_rms_labels_sorted,npf_rms_labels_sorted,cpf_rms_labels_sorted))
combine_even = np.linspace(1,49,49)
arguments_rms = np.argsort(combine_rms)
value_rms = np.sort(combine_rms)
sorted_labels = []
for n in range(0, len(arguments_rms)):
               sorted_labels.append(combine_label[arguments_rms[n]])

plt.plot(combine_even, value_rms)
plt.xticks(combine_even, sorted_labels, rotation=90)
plt.show()
fig, ax1 = plt.subplots(1,1)

c_track_even = np.linspace(1,25,25)
c_track_mean = np.mean(np.sum(cpf_w,axis=2),axis=0)
#sort_list_cpf = np.argsort(c_track_mean)[:]
#c_track_mean_sort = c_track_mean[sort_list_cpf]
#c_track_std_sort = np.std(np.sum(cpf_w,axis=2),axis=0)[sort_list_cpf]
c_track_std = np.std(np.sum(cpf_w,axis=2),axis=0)
plt.errorbar(c_track_even,c_track_mean,c_track_std)
plt.xlabel('Track number')
plt.title('Charged tracks LRP')
plt.show()

npf_track_even = np.linspace(1,25,25)
npf_track_mean = np.mean(np.sum(npf_w,axis=2),axis=0)
npf_track_std = np.std(np.sum(npf_w,axis=2),axis=0)
plt.errorbar(npf_track_even,npf_track_mean,npf_track_std)
plt.xlabel('Track number')
plt.title('Neutral tracks LRP')
plt.show()


sv_even = np.linspace(1,4,4)
sv_mean = np.mean(np.sum(sv_w,axis=2),axis=0)
sv_std = np.std(np.sum(sv_w,axis=2),axis=0)
plt.errorbar(sv_even,sv_mean,sv_std)
plt.show()

