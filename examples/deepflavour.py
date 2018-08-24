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

x_global = np.load('DataStuff/NumpyConversion/Dir_features_0.npy')
x_cpf = np.load('DataStuff/NumpyConversion/Dir_features_1.npy')
x_npf = np.load('DataStuff/NumpyConversion/Dir_features_2.npy')
x_sv = np.load('DataStuff/NumpyConversion/Dir_features_3.npy')
x_reg = np.load('DataStuff/NumpyConversion/Dir_features_4.npy')
means = np.load('DataStuff/NumpyConversion/Dir_meansandnorms.npy')
w0 = np.load('DataStuff/NumpyConversion/Dir_weights_0.npy')
w1 = np.load('DataStuff/NumpyConversion/Dir_weights_1.npy')
y0 = np.load('DataStuff/NumpyConversion/Dir_truth_0.npy')
y1 = np.load('DataStuff/NumpyConversion/Dir_truth_1.npy') 
blab=load_model('DataStuff/DeepFlavour_noRNN_nosoftmax.h5', custom_objects=global_loss_list)
blab.summary()
#blab.layers.pop()
#blab.layers.pop()
#blab.layers.pop()
#blab.outputs = [blab.layers[-1].output]
#blab.layers[-1].outbound_nodes = []
#blab.layers[-1].activation = None
#blab.inputs = blab.inputs[:4]
#apply_modifications(blab,custom_objects=global_loss_list)
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
                #analyzer = innvestigate.create_analyzer("lrp.sequential_preset_b", blab, epsilon = 1)
                analyzer = innvestigate.create_analyzer("gradient", blab)
                analysis = analyzer.analyze(inputs)
                if firsttime:
                        global_w = np.abs(analysis[0])
                        cpf_w = np.abs(analysis[1])
                        npf_w = np.abs(analysis[2])
                        sv_w = np.abs(analysis[3])
                        firsttime = False
                else:
                        global_w = np.concatenate([global_w,np.abs(analysis[0])])
                        cpf_w = np.concatenate([cpf_w,np.abs(analysis[1])])
                        npf_w = np.concatenate([npf_w,np.abs(analysis[2])])
                        sv_w = np.concatenate([sv_w,np.abs(analysis[3])])
                a = a+300
        b = b+300

np.save("lrpweights/global_grad_weights.npy",global_w)
np.save("lrpweights/cpf_grad_weights.npy",cpf_w)
np.save("lrpweights/npf_grad_weights.npy",npf_w)
np.save("lrpweights/sv_grad_weights.npy",sv_w)
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
#cpf_w = np.sum(cpf_w,axis=1)
#cpf_mean = np.mean(cpf_w,axis=0)
#cpf_w_rms = rms(cpf_w)
#cpf_rms_args = np.argsort(cpf_w_rms)
#cpf_w_rms_sort = cpf_w_rms[cpf_rms_args[:]]
#cpf_arguments = np.argsort(cpf_mean)
#cpf_std = np.std(cpf_w,axis=0)
#cpf_std_args = np.argsort(cpf_std)
#cpf_mean_sorted = cpf_mean[cpf_arguments[:]]
#cpf_std_sorted = cpf_std[cpf_arguments[:]]
#cpf_mean_std_sorted = cpf_mean[cpf_std_args[:]]
#cpf_std_std_sorted = cpf_std[cpf_std_args[:]]
#cpf_covariance = np.cov(cpf_w.transpose())
#cpf_kong = []
#cpf_std_kong = []
#cpf_rms_kong = []
#for n in range(0, len(cpf_arguments)):
#        cpf_kong.append(cpf_labels[cpf_arguments[n]])
#        cpf_std_kong.append(cpf_labels[cpf_std_args[n]])
#        cpf_rms_kong.append(cpf_labels[cpf_rms_args[n]])

cpf_even = np.linspace(1,16,16)

cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_kong, cpf_std_kong, cpf_rms_kong, cpf_w_rms_sort = plot_god(cpf_w,cpf_labels)


plt.errorbar(cpf_even,cpf_mean_sorted,cpf_std_sorted)
plt.xticks(cpf_even, cpf_kong, rotation=90)
plt.show()
plt.errorbar(cpf_even,cpf_mean_std_sorted,cpf_std_std_sorted)
plt.xticks(cpf_even, cpf_std_kong, rotation=90)
plt.show()
plt.plot(cpf_even,cpf_w_rms_sort)
plt.xticks(cpf_even, cpf_rms_kong, rotation=90)
plt.show()

npf_labels = ['Npfcan_ptrel',
              'Npfcan_deltaR',
              'Npfcan_isGamma',
              'Npfcan_HadFrac',
              'Npfcan_drminsv',
              'Npfcan_puppiw'
]

npf_even = np.linspace(1,6,6)

npf_mean_sorted, npf_std_sorted, npf_mean_std_sorted, npf_std_std_sorted, npf_kong, npf_std_kong, npf_rms_kong, npf_w_rms_sort = plot_god(npf_w,npf_labels)


plt.errorbar(npf_even,npf_mean_sorted,npf_std_sorted)
plt.xticks(npf_even, npf_kong, rotation=90)
plt.show()
plt.errorbar(npf_even,npf_mean_std_sorted,npf_std_std_sorted)
plt.xticks(npf_even, npf_std_kong, rotation=90)
plt.show()
plt.plot(npf_even,npf_w_rms_sort)
plt.xticks(npf_even, npf_rms_kong, rotation=90)
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
sv_mean_sorted, sv_std_sorted, sv_mean_std_sorted, sv_std_std_sorted, sv_kong, sv_std_kong, sv_rms_kong, sv_w_rms_sort = plot_god(sv_w,sv_labels)

plt.errorbar(sv_even,sv_mean_sorted,sv_std_sorted)
plt.xticks(sv_even, sv_kong, rotation=90)
plt.show()
plt.errorbar(sv_even,sv_mean_std_sorted,sv_std_std_sorted)
plt.xticks(sv_even, sv_std_kong, rotation=90)
plt.show()
plt.plot(sv_even,sv_w_rms_sort)
plt.xticks(sv_even, sv_rms_kong, rotation=90)
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
global_mean_sorted, global_std_sorted, global_mean_std_sorted, global_std_std_sorted, global_kong, global_std_kong, global_rms_kong, global_w_rms_sort = plot_god2(global_w,global_labels)

plt.errorbar(global_even,global_mean_sorted,global_std_sorted)
plt.xticks(global_even, global_kong, rotation=90)
plt.show()
plt.errorbar(global_even,global_mean_std_sorted,global_std_std_sorted)
plt.xticks(global_even, global_std_kong, rotation=90)
plt.show()
plt.plot(global_even,global_w_rms_sort)
plt.xticks(global_even, global_rms_kong, rotation=90)
plt.show()

combine_rms = np.concatenate((global_w_rms_sort,sv_w_rms_sort,npf_w_rms_sort,cpf_w_rms_sort))
combine_label = np.concatenate((global_rms_kong,sv_rms_kong,npf_rms_kong,cpf_rms_kong))
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

hist_global_w = []
hist_cpf_w = []
hist_npf_w = []
hist_sv_w = []
longlist = []
totsum = 0
large_sum = 0
for n in range(0,global_w.shape[0]):
    b1 = global_w[n]
    b2 = cpf_w[n].flatten()
    b3 = npf_w[n].flatten()
    b4 = sv_w[n].flatten()
    kop = np.append(b1,b2)
    po = np.append(b3,b4)
    koppo = np.append(kop,po)
    large_sum = large_sum + koppo
    a1 = global_w[n]
    a2 = sum(cpf_w[n])
    a3 = sum(npf_w[n])
    a4 = sum(sv_w[n])
    kap = np.append(a1,a2)
    pa = np.append(a3,a4)
    kappa = np.append(kap,pa)
    longlist.append(kappa)
    totsum = totsum + kappa


arguments = np.argsort(totsum)
value = np.sort(totsum)

mega_arguments = np.argsort(large_sum)
mega_value = np.sort(large_sum)


cpf_nr_labels = []


for n in range(1,26):
    for z in range(0,len(cpf_labels)):
        cpf_nr_labels.append('track'+str(n)+'_'+cpf_labels[z])

npf_nr_labels = []

for n in range(1,26):
    for z in range(0,len(npf_labels)):
        npf_nr_labels.append('track'+str(n)+'_'+npf_labels[z])



sv_nr_labels = []
for n in range(1,5):
    for z in range(0,len(sv_labels)):
        sv_nr_labels.append('track'+str(n)+'_'+sv_labels[z])



labels = global_labels+cpf_labels+npf_labels+sv_labels

mega_labels = global_labels+cpf_nr_labels+npf_nr_labels+sv_nr_labels

kong = []
for n in range(0, len(arguments)):
               kong.append(labels[arguments[n]])

king_kong = []
for n in range(0, len(mega_arguments)):
               king_kong.append(mega_labels[mega_arguments[n]])

               
#plt.hist(hist_global_w, bins=15, range=(-0.5,14.5))
#plt.show()
#plt.hist(hist_cpf_w, bins=16, range=(-0.5,15.5))
#plt.show()
#plt.hist(hist_npf_w, bins=6, range=(-0.5,5.5))
#plt.show()
#plt.hist(hist_sv_w, bins=12, range=(-0.5,11.5))
#plt.show()


plt.plot(np.sum(sum(cpf_w),axis=1))
plt.title('charged candidates')
plt.xlabel('track nr')
plt.show()
plt.plot(np.sum(sum(npf_w),axis=1))
plt.title('neutral candidates')
plt.xlabel('track nr')
plt.show()
plt.plot(np.sum(sum(sv_w),axis=1))
plt.title('secondary vertex')
plt.xlabel('vertex nr')
plt.show()
even = np.linspace(1,49,49)
plt.plot(even,value)
plt.xticks(even, kong, rotation=90)
plt.show()
