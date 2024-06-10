# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:18:22 2020

@author: Arash


IMPROVEMENTS PLANNED:
    Interpolation for missing shifts.
    add BL Correction: iterative smoothing-splines with root error adjustment (ISREA) algorithm Y. Xu, P. Du, R. Senger, J. Robertson, and J. L. Pirkle, \ISREA: An Ecient Peak-Preserving Baseline Correction Algorithm for Raman Spectra," Applied Spectroscopy, vol. 75, no. 1, pp. 34{45, 2021. 
    add BL Correction: iterative polyfit method from Lieber and Mahadevan-Jansen: C. A. Lieber and A. Mahadevan-Jansen, \Automated method for subtraction of fluorescence from biological raman spectra," Applied spectroscopy, vol. 57, no. 11, pp. 1363{1367, 2003.
    add smoothing techniques moving average (MA), moving median (MM), Gaussian moving average (GMA) and locally weighted scatter plot smoothing (LOWESS).
    add outlier detection: nearest neighbors but specically to the median spectrum
    add classification: Discriminant analysis, LDA, PCA, t-SNE
    Check the noise characterization page 25 section 4.2
    plt.suptitle('') # super title
"""

#%%imports
"""IMPORTS"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.ensemble import IsolationForest
from pandas import DataFrame, concat, read_feather, unique #ExcelWriter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from scipy.sparse import linalg
#import spectrochempy as scp
import os
from tqdm import tqdm #to make a progress bar
from math import factorial
import pywt
from matplotlib.ticker import AutoMinorLocator
import random




#%%initiation
"""Parameters"""
mainDirectory ='C:\\Users\\mfeizpou\\OneDrive - Vrije Universiteit Brussel\\Desktop\\31-07-2023 CSF Bacteria Salma' #remember the two slashes!
os.chdir(mainDirectory)
if not os.path.exists('Figures'):
    os.makedirs('Figures')

# load the data as df
dataName = 'Data-unprocessed'
datalimMIN = 0
#datalimMAX = 10
data = read_feather(dataName)#.iloc[datalimMIN:10] #selecting a small subset to make trials faster
data = data.reset_index(drop=True)
shifts = read_feather(dataName+'_shifts')


infoNumCol = 1 # number of non-data columns
normType = 'z-score' # linear or z-score normalization 
scoreType = 'Whitaker-Hayes' # for Spike Removal
RamanThreshold = 10 # for Spike Removal
#FTIRThreshold = 4 * RamanThreshold # for Spike Removal

#BL
"""
We found that generally 0.001 ≤ p ≤ 0.1 is a good choice 
(for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9 , 
"""
assym = 0.01; assymRange = (-3, -1) # 10 to power
lamb = 1e7; lamRange = (2,9) # 10 to power
blRes = 10 # optim res grid
blMethod = 'als' # polynomial, als, arpls
polOrder = 11

#outlier detection
oLevel = 0 # It depends on the dataset but outliers are very uncertain -> their z-score is higher
oFactor = 0.5 # if more than 50% of the spectrum is above the selected threshold, the spectrum is removed

"""Switches"""
olSwitch = bool(1)
olPlotSwitchBool = bool(1)
# #FTIR
# # spikeSwitchFTIRBool = bool(0)
# BLSwitchFTIRBool = bool(1)
# normSwitchFTIRBool = bool(1)
# olSwitchFTIRBool = bool(1)
#Raman
spikeSwitchRamanBool = bool(1)
BLSwitchRamanBool = bool(1)
blPlotSwitch = bool(1)
normSwitchRamanBool = bool(1)
olSwitchRamanBool = bool(1)


#spike removal
spr_method = 'wh_collective' #options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
# mr
wname = 'db1' # wavelet name for DWT

# #separators
# FTIRcolumnNames = [idx for idx in data.columns if idx[-1] == 'f']
# FTIRSeparatorIndex = data.columns.get_loc(FTIRcolumnNames[0])
# shiftIndexFTIR = FTIRSeparatorIndex-infoNumCol #the "shifts" index from which FTIR shifts start

#filters
SGFilterSwitch = 1
SGPlotSwitch = 1            
sgWin = 11; sgWinRange = (5,15) # Savitzky Golay window size
sgOrder = 5; sgOrderRange = (3,10) # Savitzky Golay order of polynomial to be fitted
sgRes = 5

#%% Functions

"""Definitions"""
def z_score(intensity, scoreType): #https://www.statisticshowto.com/probability-and-statistics/z-score/
    if scoreType == 'normal':
        mean_int = np.mean(intensity)
        std_int = np.std(intensity)
        z_scores = (intensity - mean_int) / std_int
    if scoreType == 'modified':
        median_int = np.median(intensity)
        mad_int = np.median([np.abs(intensity - median_int)])
        z_scores = 0.6745 * (intensity - median_int) / mad_int # 0.6745 is the 0.75th quartile of the standard normal distribution, to which the MAD converges to
    if scoreType == 'Whitaker-Hayes':
        # First we calculated ∇x(i):
        delta_int = np.diff(intensity)    
        # Calculation of the z scores of the gradient of the spectrum and its plot:
        median_int = np.median(delta_int)
        mad_int = np.median([np.abs(delta_int - median_int)])
        z_scores = 0.6745 * (delta_int - median_int) / mad_int # 0.6745 is the 0.75th quartile of the standard normal distribution, to which the MAD converges to
    return z_scores

def spikeRemoval(x, y, ramORftir, data_z = 0, scoreType = 'Whitaker-Hayes', m = 3, threshold = 6, spr_method = 'wh_collective', userconfSwitch='n'): # Whitaker, Darren A., and Kevin Hayes. “A simple algorithm for despiking Raman spectra.” Chemometrics and Intelligent Laboratory Systems 179 (2018): 82–84.
    global RamanThreshold
    global i
    if spr_method == 'mr': # Maury-Revilla
        y_out = mr_spikeRemoval(x, y, ramORftir, thresLevel = RamanThreshold, wname = 'db1', m = m)
    elif spr_method == 'wh_collective':
        y_out = spikeRemoval_collective(x, y, data_z, ramORftir, scoreType = 'Whitaker-Hayes', m = m, threshold = RamanThreshold, userconfSwitch=userconfSwitch)
    elif spr_method == 'wh':
        intensity_z_score = z_score(y.values, scoreType = scoreType) # scoreType can be 'modified' or 'normal' or 'Whitaker-Hayes'
        #spikePlot(x.values[0][1:], intensity_z_score, ramORftir, threshold) # make a plot of the spectrum and the threshold
        spikes = abs(intensity_z_score) > threshold # 1 is assigned to "suspected" spikes, 0 to non-spikes
        y_out = y.copy() # So we don't overwrite y
        for spe in np.arange(len(spikes)):
            if spikes[spe] != 0: # If we have an spike in position i
                # defining a window for interpolation considering the beg/end indices
                if spe-m < 0:   
                    w = np.arange(0, spe+1+m) # we select 2 m + 1 points around our spike
                elif spe+1+m > len(y)-1:
                    w = np.arange(spe-m, len(y)-1) # we select 2 m + 1 points around our spike
                else:
                    w = np.arange(spe-m, spe+1+m) # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                y_out[spe] = np.mean(y[w2]) # and we average the value
        if userconfSwitch == 'y':
            #getting user confirmation
            if sum(spikes) > 0:
                plt.figure()
                plt.suptitle( 'Spike Removal - User Confirmation Needed - sample {}'.format(i))
                plt.subplot(1, 2, 1) # row 1, col 2 index 1
                plt.plot(x.values.reshape(y.shape[0],), y)
                plt.title(ramORftir + ' Spectrum')
                plt.xlabel('Shifts (cm-1)')
                plt.ylabel('Modified Intensity (a.u.)')
                
                
                plt.subplot(1, 2, 2) # row 1, col 2 index 2
                plt.plot(x.values[1:].reshape(intensity_z_score.shape[0],), abs(intensity_z_score))
                plt.title('Z-score Plot')
                plt.xlabel('Shifts (cm-1)')
                plt.ylabel('Z-Score (cm-1)')
                plt.hlines(threshold, x.values[0], x.values[-1], linestyles='-', color = 'red')
                plt.tight_layout()
                plt.show() 
            
                userconf = input('Is the spike correctly selected? options: y (yes),n (no, renews selection with new threshold), and ns (no spike, undoes the changes):\n')
                if userconf.lower() == 'n':
                    newthres = float(input('Current threshold is {}. Please provide a new suitable threshold:\n'.format(threshold)))
                    RamanThreshold = newthres
                    spikes = abs(intensity_z_score) > threshold # 1 is assigned to "suspected" spikes, 0 to non-spikes
                    y_out = y.copy() # So we don't overwrite y
                    for spe in np.arange(len(spikes)):
                        if spikes[spe] != 0: # If we have an spike in position i
                            # using w from before
                            # w = np.arange(spe-m,spe+1+m) # we select 2 m + 1 points around our spike
                            w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                            y_out[spe] = np.mean(y[w2]) # and we average the value
                elif userconf.lower() == 'ns':
                    newthres = float(input('Current threshold is {}. Please provide a new suitable threshold:\n'.format(threshold)))
                    RamanThreshold = newthres
                    y_out = y
                elif userconf.lower() == 'y':
                    pass
                elif userconf.lower() == '':
                    pass
                else:
                    raise ValueError('The options you can choose from are y (yes),n (no), and ns (no spike).')
    return y_out

def spikeRemoval_collective(x, y, data_z, ramORftir, scoreType = 'Whitaker-Hayes', m = 3, threshold = 6, userconfSwitch='n'):
    global RamanThreshold
    global i
    intensity_z_score = data_z.loc[i] # scoreType can be 'modified' or 'normal' or 'Whitaker-Hayes'
    #spikePlot(x.values[0][1:], intensity_z_score, ramORftir, threshold) # make a plot of the spectrum and the threshold
    spikes = abs(intensity_z_score) > threshold # 1 is assigned to "suspected" spikes, 0 to non-spikes
    y_out = y.copy(deep=True) # So we don't overwrite y
    for spe in np.arange(len(spikes)):
        if spikes[spe] != 0: # If we have an spike in position spe
            # defining a window for interpolation considering the beg/end indices
            if spe-m < 0:   
                w = np.arange(0, spe+1+m) # we select 2 m + 1 points around our spike
            elif spe+1+m > len(y)-1:
                w = np.arange(spe-m, len(y)-1) # we select 2 m + 1 points around our spike
            else:
                w = np.arange(spe-m, spe+1+m) # we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
            y_out[spe] = np.mean(y[w2]) # and we average the value
    
    if userconfSwitch == 'y':    
        #getting user confirmation
        if sum(spikes) > 0:
            plt.figure()
            plt.suptitle( 'Spike Removal - User Confirmation Needed - sample {}'.format(i))
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            plt.plot(x.values.reshape(y.shape[0],), y)
            plt.title(ramORftir + ' Spectrum')
            plt.xlabel('Shifts (cm-1)')
            plt.ylabel('Modified Intensity (a.u.)')
            
            
            plt.subplot(1, 2, 2) # row 1, col 2 index 2
            plt.plot(x.values.reshape(intensity_z_score.shape[0],), abs(intensity_z_score))
            plt.title('Z-score Plot')
            plt.xlabel('Shifts (cm-1)')
            plt.ylabel('Z-Score (cm-1)')
            plt.hlines(threshold, x.values[0], x.values[-1], linestyles='-', color = 'red')
            plt.tight_layout()
            plt.show() 
            
            userconf = input('Is the spike correctly selected? options: y (yes),n (no), and ns (no spike):\n')
            if userconf.lower() == 'n':
                newthres = input('Current threshold is {}. Please provide a new suitable threshold:\n'.format(threshold))
                if newthres == '':
                    newthres = RamanThreshold #if you press enter, then same threshold applies
                RamanThreshold = float(newthres)
                spikes = abs(intensity_z_score) > threshold # 1 is assigned to "suspected" spikes, 0 to non-spikes
                y_out = y.copy() # So we don't overwrite y
                for spe in np.arange(len(spikes)):
                    if spikes[spe] != 0: # If we have an spike in position i
                        # using w from before
                        # w = np.arange(spe-m,spe+1+m) # we select 2 m + 1 points around our spike
                        w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                        y_out[spe] = np.mean(y[w2]) # and we average the value
            elif userconf.lower() == 'ns':
                newthres = input('Current threshold is {}. Please provide a new suitable threshold:\n'.format(threshold))
                if newthres == '':
                    newthres = RamanThreshold #if you press enter, then same threshold applies
                RamanThreshold = float(newthres)
                y_out = y
            elif userconf.lower() == 'y':
                pass
            elif userconf.lower() == '':
                pass
            else:
                raise ValueError('The options you can choose from are y (yes),n (no), and ns (no spike).')
    return y_out

def mr_spikeRemoval(shifts_, intensities, ramORftir, thresLevel = 3, wname = 'db1', m = 3): #https://journals.sagepub.com/doi/abs/10.1366/14-07834
    """"Maury Revilla spike removal 2015 
    The idea is to """
    intensities_diff = np.diff(intensities) # formula 2 in the paper - D
    #note: np.diff makes the vector one element shorter
    N = len(intensities_diff) # length of spectrum
    
    # calculating Allan deviation = ADEV
    acum = 0 # for the summation
    for ci in np.arange(N-1): # summation
        acum += (intensities_diff[ci+1] - intensities_diff[ci])**2
    adev = np.sqrt(acum / (2*(N - 2))) # formula 6 in the paper
    threshold = thresLevel * adev # as in the case of Z-score, we need a threshold
    
    [intensities_diff_dwt, _] = pywt.dwt(intensities_diff, wname) # wname: wavelet name, returns approximation coefficients vector and detail coefficients vector
    #note: dwt makes the vector one element shorter
    intensities_diff_filtered = pywt.idwt(intensities_diff_dwt, None, wname) # filtered D, we are losing the detail coefficients vector by passing none instead
    intensities_diff_filtered = intensities_diff_filtered[:N]
    deltas = intensities_diff - intensities_diff_filtered #residuals - formula 7 of the paper
    # Tag index if delta is outside the threshold
    tags = abs(concat([DataFrame(deltas), DataFrame([0])], axis = 0).values) > abs(threshold) # Fig 3.c of the paper, concat so tags can be used as the index of intensities (1015)
    
    #Tag y(i+-m) (m = 1, 2, ...) if yi is already tagged and Di is outside the threshold.
    for ci in np.arange(N):
        if tags[ci]:
            for jj in np.arange(max(1, ci - m), min(ci + m, N)):
                if ci == jj:
                    continue
                if abs(intensities_diff[jj]) > abs(threshold): # is Di outside the threshold? if so, tag the index
                    tags[jj] = True
    tags = tags.reshape((len(tags))) # changing the shape so this array can be passed as the index of intensities
    # tags_final = np.zeros(len(tags))
    # for ci in np.arange(2, len(tags)-1):
    #         if tags[ci - 1] and tags[ci + 1]: # Tag yi if both yi–1 and yi+1 are already tagged
    #             tags_final[ci] = 1
    # tags_final = tags_final == 1
    if any(tags):        
        spikes = np.zeros((len(tags),))
        spikes[tags] = intensities[tags]
        
        intensities[tags] = [np.NaN] * len(tags)
        corrected_spectrum = DataFrame(intensities, dtype=float).interpolate().values.ravel() #interpolation happens here
    else:
        corrected_spectrum = intensities
    return(corrected_spectrum)
    

def spikePlot(x, y, ramORftir, threshold):
    plt.figure()
    plt.plot(x.reshape(y.shape[0],), y)
    plt.plot(x.reshape(y.shape[0],), threshold*np.ones(len(x)), label = 'threshold') # Line showing the threshold
    plt.title(ramORftir + ' Z-Score - Row {}'.format(i))
    plt.xlabel( 'Shifts (cm-1)')
    plt.ylabel( '|Z-Score| (a.u.)')
    plt.legend()#loc = 'upper left' )

def normalizer(specVec, normType):    
    specVec = specVec.reshape(-1, 1) #going from (X,) to (X,1)
    if normType == 'linear':
        scaler = MinMaxScaler() #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        scaler.fit(specVec)
        specVec_normalized = scaler.transform(specVec) 
    elif normType == 'z-score':
        scaler = StandardScaler()
        scaler.fit(specVec)
        specVec_normalized = scaler.transform(specVec)
    else: 
        raise ValueError('The chosen normalizer does not exist. Please choose \'linear\' or \'z-score\'.')
    return(specVec_normalized)


def baselineRemoval(y, xrs, blMethod, plotBool, lam = 1e7,
                 p = 0.1, order = 7, niter=200, label = 'Original Data', pt = 'Unknown'):
    
    if blMethod == 'als':
        bl = baseline_als(y, lam = lam, 
                          p = p, niter=niter, label = 'Raman')
    elif blMethod == 'arpls':
        bl = baseline_arPLS(y, ratio=p, lam=lam, niter=niter)
    # elif blMethod == 'ipoly': # https://www.spectrochempy.fr/latest/userguide/processing/baseline.html#Advanced-baseline-correction
    #     # #https://www.spectrochempy.fr/latest/userguide/dataset/dataset.html
    #     # #making an appropriate to scp dataset
    #     # d1D = scp.NDDataset(y, dims=['x'])
    #     # d1D.x = scp.Coord(xrs)
    #     # d1D.x.title = 'Raman Shifts'
    #     # d1D.x.units = 'cm^-1'
        
    #     # ranges = [[600.,620.],[1010.,1070.],[1700.,1705.]]
    #     # blc = scp.BaselineCorrection(d1D)
    #     # blc.compute(*ranges, interpolation="polynomial", method='sequential', order=order).data
    #     # bl = y - blc.corrected.data
    #     bl = iter_polyfit(y, degree=11, n_iter=niter)
    
    if plotBool: #
        #print('i = {}; first index: {}'.format(i, data[data['Class'] == pt].index[0]))
        currdir = os.curdir
        if data[data['Class'] == pt].index[0] == i:
            os.chdir(mainDirectory + '\\Figures')   
            if not os.path.exists('Baseline and Spectrum Plots'):
                os.makedirs('Baseline and Spectrum Plots')
            os.chdir(mainDirectory + '\\Figures\\Baseline and Spectrum Plots')
            if not os.path.exists(pt):
                os.makedirs(pt)
        os.chdir(mainDirectory + '\\Figures\\Baseline and Spectrum Plots\\' + pt)
        if data[data['Class'] == pt].index[0] < i < data[data['Class'] == pt].index[0] + 10:
    
            plt.figure()
            plt.ylabel('Intensity (a.u.)')
            plt.xlabel('Raman Shift (cm-1)')
            plt.title(label + ' Spectrum and {} Baseline - {} - Sample {}'.format(blMethod, pt, i))
            plt.plot(xrs.reshape(y.shape[0],), y, label = 'Spectrum')
            plt.plot(xrs.reshape(y.shape[0],), bl, label = 'Baseline') # z is the baseline
            plt.legend()
            
            #Saving the figure
            figName = "Spectrum AND {} BaseLine - {}".format(blMethod, pt) + '_file_' + str(i) #this i should be the same for all the algorithms!
            plt.savefig(figName, dpi=300, bbox_inches = "tight")
            plt.close()           
            
            plt.figure()
            plt.ylabel('Intensity (a.u.)')
            plt.xlabel('Raman Shift (cm-1)')
            plt.title(label + ' - {} Baseline-Corrected Spectrum - {} - Sample {}'.format(blMethod, pt, i))
            plt.plot(xrs.reshape(y.shape[0],), y-bl, label = 'Baseline-Corrected Spectrum') # z is the baseline
                        
            #Saving the figure
            figName = "{} BaseLine-Corrected Spectrum - {}".format(blMethod, pt) + '_file_' + str(i) #this i should be the same for all the algorithms!
            plt.savefig(figName, dpi=300, bbox_inches = "tight")
            plt.close()
            os.chdir(currdir)
            
    return(bl)




def BLOptimizer(y, xrs, blMethod, lam = lamRange,
                 p = assymRange, blRes=3, niter=200, label = 'Original Data', pt = 'Unknown'):
    if blMethod == 'als':
        lRange = np.arange(lamRange[0], lamRange[1], (lamRange[1]-lamRange[0])/blRes)
        pRange = np.arange(assymRange[0], assymRange[1], (assymRange[1]-assymRange[0])/blRes)
        fig, axs = plt.subplots(blRes, blRes)
        for xil, lam in enumerate(lRange):
            for xip, p in enumerate(pRange): 
                bl = baseline_als(y, lam = 10.**lam, 
                                  p = 10.**p, niter=niter,
                                  label = 'Raman')
                axs[xil, xip].plot(xrs.reshape(y.shape[0],), bl, linewidth=0.5, label = 'Baseline') # z is the baseline
                axs[xil, xip].plot(xrs.reshape(y.shape[0],), y, alpha=0.7, linewidth=0.5, label = 'Spectrum')
                axs[xil, xip].set(xlabel=str(round(p,1)), ylabel=str(round(lam,1)))
        for ax in axs.flat:
            ax.label_outer()
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.supylabel('lambda') 
    fig.supxlabel('assymetry parameter (p)')       
    # Saving the figure
    os.chdir(mainDirectory + '\\Figures')   
    if not os.path.exists('Baseline and Spectrum Plots'):
        os.makedirs('Baseline and Spectrum Plots')
    os.chdir(mainDirectory + '\\Figures\\Baseline and Spectrum Plots')
    figName = "BL Correction Paraemeter Choice - " + blMethod + '.jpg'
    plt.savefig(figName, dpi=blRes*100, bbox_inches = "tight")
    plt.show() 
    
    lam = input('Please choose the best lambda based on the figure:\n')
    p = input('Please choose the best assymetry parameter (p) based on the figure:\n')
    
    return(10.**float(lam),10.**float(p))
        
        
        
        
        
# def iter_polyfit(spectrum, degree=11, n_iter=100):
#     input_spectrum = spectrum;
#     [first, last] = find_nonzeros(spectrum);
       
#     if first == -1 && last == -1
#         corrected_spectrum = input_spectrum;
#         base = zeros(length(spectrum(:, 1)), 1);
#     else
#         spectrum = spectrum(first:last, :);
    
#         shifts = spectrum(:, 1);
#         intensities = spectrum(:, 2);
#         iter = 0;
               
#         while iter < n_iter
#             p = polyfit(shifts, intensities, degree);
#             poly_data = polyval(p, shifts);
     
#             if all(intensities <= poly_data)
#                 break
#             end
     
#             intensities = min(intensities, poly_data);
#             iter = iter + 1;
#         end
    
#         base = zeros(length(input_spectrum(:, 1)), 1);
#         base(first:last) = poly_data;
#         aux = [shifts spectrum(:, 2) - poly_data];
#         input_spectrum(first:last, :) = aux;
#         corrected_spectrum = input_spectrum;
    


def baseline_als(y, lam = 1e5,
                 p = 0.1, niter=200, label = 'Original Data'):
    """
    There are two parameters: p for asymmetry and λ for smoothness. 
    Both have to be tuned to the data at hand. 
    We found that generally 0.001 ≤ p ≤ 0.1 is a good choice 
    (for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9 , 
    but exceptions may occur. In any case one should vary λ on 
    a grid that is approximately linear for log λ
    """
    global i
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    
    for ppp in range(niter):
      W = sparse.spdiags(w, 0, L, L) #https://het.as.utexas.edu/HET/Software/Scipy/generated/scipy.sparse.spdiags.html#:~:text=spdiags,-scipy.sparse.spdiags&text=Return%20a%20sparse%20matrix%20from%20diagonals.&text=By%20default%20(format%3DNone),sparse%20matrix%20format%20is%20returned.
      Z = W + lam * D.dot(D.transpose())
      z = spsolve(Z, w*y.astype(np.float64)) #the baseline
      w = p * (y > z) + (1-p) * (y < z)
      #print(y)            
    return z

def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z

def outlierDetector(x, Y, oLevel, plotSwitch, ramORftir, plasT):
    """
        :deleteOutliersSummary: Calls the z score function to get   
        the outlier spectra numbers.We are interested in x 
        coordinate of the results. In our case, the x coordinate is 
        the spectra number.So, we apply the counter function on the 
        result to get the total count of outlier points for 
        spectra.and delete the spectra if the spectra has 25% of 
        its points as outliers
        :param X: Training spectral file (usually MSCTrain)
        :type X: array
        :param Y: Training target file      
        :type Y: array
        :returns: individualX (list) and y (list), 
        New spectral & target train files with outliers eliminated
    """
    #Making a Z-Score holder
    Y_Z = Y.copy(deep=True) #so changes to X are not reflected in X_Z
    
    # deleteSpectra stores the Spectra number > 75% points as outliers
    deleteSpectra = []
    featureCount = Y.shape[1]
    for fe in range(0,featureCount):
        # call the function
        featureVec = Y.iloc[:,fe]
        featureVec_Z = z_score(featureVec, scoreType='normal')
        # print sample number and spectra number with its   
        #corresponding number of outlier points
        Y_Z.iloc[:,fe] = featureVec_Z
        
        
        
    # """Compare the Z-score plots for a normal and abnormal case and use their difference as the detector"""
    
    # plt.figure()
    # plt.title('sample 0 abnormal - Spec')
    # plt.plot(x.values.reshape(Y.shape[1],), Y.iloc[0][:])
    # plt.figure()
    # plt.title('sample 0 abnormal - Z-score avg = {}'.format(np.mean(Y_Z.iloc[0][:])))
    # plt.plot(x.values.reshape(Y.shape[1],), Y_Z.iloc[0][:])
    
    # plt.figure()
    # plt.title('sample 62 normal')
    # plt.plot(x.values.reshape(Y.shape[1],), Y.iloc[62][:])
    # plt.figure()
    # plt.title('sample 62 normal - Z-score avg = {}'.format(np.mean(Y_Z.iloc[62][:])))
    # plt.plot(x.values.reshape(Y.shape[1],), Y_Z.iloc[62][:])
    
    
    
    
    for sa in np.arange(Y.shape[0]):
        # sum(int(featureVec_Z[sa,:] > threshold)) is the number of anamolous points
        if sum(abs(Y_Z.iloc[sa,:]) > oLevel) > oFactor * Y_Z.shape[1]: #if the number of anamolous points are more than 30% of total number of features
            if plotSwitch:
                os.chdir(mainDirectory + '\\Figures')
                if not os.path.exists('Outliers'):
                    os.makedirs('Outliers')
                os.chdir(mainDirectory + '\\Figures' + "\\Outliers")
    
                plt.figure()
                plt.plot(x.values.reshape(Y.shape[1],), Y.iloc[sa][:])
                plt.title(ramORftir + ' - ' + plasT + ' - Outliers sample index: {}'.format(sa))
                plt.xlabel('Shifts (cm-1)')
                plt.ylabel('Modified Intensity (cm-1)')
                
                figName = "outlier_" + ramORftir + '_' + plasT + '_sample_' + str(sa) #this i should be the same for all the algorithms!
                plt.savefig(figName, dpi=300, bbox_inches = "tight")
                os.chdir(mainDirectory)
                plt.close()                
            
            deleteSpectra.append(Y.index[sa]) # df.drop() works with the labels themselves
    
    #returning the indices that have to be removed
    return(Y_Z, deleteSpectra)

def savitzky_golay(y, window_size, order, deriv=0, rate=10):
    #ref: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
                             
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**tt for tt in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# def KalmanFilt(z, Q, R, numRepetitions, numSpectra):    
#     # allocate space for arrays
#     xhat=np.zeros(numRepetitions)      # a posteri estimate of x
#     P=np.zeros(numRepetitions)         # a posteri error estimate
#     xhatminus=np.zeros(numRepetitions) # a priori estimate of x
#     Pminus=np.zeros(numRepetitions)    # a priori error estimate
#     K=np.zeros(numRepetitions)         # gain or blending factor

#     # intial guesses
#     xhat[0] = 0.0
#     P[0] = 1.0
    
#     for k in range(1,numSpectra):
#         # time update
#         xhatminus[k] = xhat[k-1]
#         Pminus[k] = P[k-1]+Q
    
#         # measurement update
#         K[k] = Pminus[k]/( Pminus[k]+R )
#         xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
#         P[k] = (1-K[k])*Pminus[k]
        
#     return(xhat[-1])



def SGOptimizer(y, xrs, sgWinRange, sgOrderRange, deriv=0, rate=10):
    if (sgOrderRange[1]-sgOrderRange[0])//sgRes == 0 or (sgWinRange[1]-sgWinRange[0])//sgRes == 0:
        raise ValueError('Adjust the resolution so that the step size is more than 1 for the SG optimizer.')
    wRange = range(sgWinRange[0], sgWinRange[1], (sgWinRange[1]-sgWinRange[0])//sgRes)
    oRange = range(sgOrderRange[0], sgOrderRange[1], (sgOrderRange[1]-sgOrderRange[0])//sgRes)
    fig, axs = plt.subplots(len(wRange), len(oRange))
    for xiw, win in enumerate(wRange):
        for xio, order in enumerate(oRange): 
            if  win > order+1:
                sg = savitzky_golay(y, win, order,
                                    deriv=deriv, rate=rate)
                axs[xiw, xio].plot(xrs.reshape(y.shape[0],), y, linewidth=0.5, label = 'UNF Spectrum') # z is the baseline
                axs[xiw, xio].plot(xrs.reshape(y.shape[0],), sg, linewidth=0.5, label = 'FIL Spectrum')
                axs[xiw, xio].set(xlabel=str(round(order,1)), ylabel=str(round(win,1)))
            else:
                axs[xiw, xio].plot(xrs.reshape(y.shape[0],), y*0, label = 'NO SG') # z is the baseline
                axs[xiw, xio].set(xlabel=str(round(order,1)), ylabel=str(round(win,1)))
    for ax in axs.flat:
        ax.label_outer()
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.supylabel('window (# data points)') 
    fig.supxlabel('polynomial order')         
    # Saving the figure
    os.chdir(mainDirectory + '\\Figures')   
    if not os.path.exists('Savitzky-Golay - Spectral Smoothing'):
        os.makedirs('Savitzky-Golay - Spectral Smoothing')
    os.chdir(mainDirectory + '\\Figures\\Savitzky-Golay - Spectral Smoothing')
    figName = "Savitzky-Golay Paraemeter Choice - " + blMethod + '.jpg'
    plt.savefig(figName, dpi=sgRes*100, bbox_inches = "tight")
    plt.show() 
    
    win = input('Please choose the best window size based on the figure:\n')
    order = input('Please choose the best polynomial order based on the figure:\n')
    
    return(int(win),int(order))
















#%% Body: Spectral Corrections     
print('\nSpectral processing in progress..\n')

# a_dummy = data.copy(deep=True)
# """ DELETE THIS """
# """ DELETE THIS """
# #data = data[data['Plastic Type'] == 'PET'].iloc[:10,:].reset_index(drop=True)
# data = data.iloc[8390:8400,:].reset_index(drop=True)
# """ DELETE THIS """
# """ DELETE THIS """


for i in tqdm(range(datalimMIN, data.shape[0])):
    pt = data['Class'][i]
    
   
    RamanSpec = data.iloc[i][infoNumCol:]
    RamanShifts = shifts.iloc[0][0:]    
  
    """ Raman BASELINE Correction """ 
    #There are two parameters: p for asymmetry and λ for smoothness. 
    #Both have to be tuned to the data at hand. 
    #We found that generally 0.001 ≤ p ≤ 0.1 is a good choice 
    #(for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9 , 
    #but exceptions may occur. In any case one should vary λ on 
    #a grid that is approximately linear for log λ
    if BLSwitchRamanBool:
        """
        Make sure to define the following if method == 'als'
        lam = 1e7, p = 0.1, niter=200
        """
        if i == 0: # using the first spectrum, the bl's parameters are optimized
            lamb, assym = BLOptimizer(y = RamanSpec.values,
                                                xrs = RamanShifts.values, blMethod = blMethod, 
                                                lam = lamRange, p = assymRange, blRes=blRes, 
                                                niter=200, label = 'Raman', pt=pt)
        RamanBLine = baselineRemoval(y = RamanSpec.values,
                                            xrs = RamanShifts.values, blMethod = blMethod, 
                                            plotBool = blPlotSwitch, lam = lamb, 
                                            p = assym, order = polOrder, niter=200, label = 'Raman', pt=pt)
        RamanSpec = RamanSpec - RamanBLine    
        #print(RamanSpec)



    """ Raman SPIKE removal """

    #done separately after this loop   




    """ Raman Normalization """
    # distance-based algorithms could really benefit from normalized data
    if normSwitchRamanBool:
        RamanSpec = normalizer(RamanSpec.values, normType)   
    
    #updated_Row = np.concatenate([RamanSpec,
    #updated_Row = RamanSpec.transpose()
    #                              FTIRSpec]).transpose()
    updated_Row = RamanSpec.transpose()
    
    data.loc[[i], data.columns[infoNumCol:]] = updated_Row
    

#%% Data Export - mid point save

print('\n\n Exporting the processed data before spike and outlier removal.')

os.chdir(mainDirectory)
dataName = dataName.replace('unprocessed', 'processed_AUTOSAVE_Normalized_BLCorrected')
data = data.reset_index(drop=True)
#data.to_excel("{}.xlsx".format(dataName)) 
data.to_feather("{}".format(dataName)) 



#%% Loading Intermediate Save
os.chdir(mainDirectory)
dataName = 'Data-processed_AUTOSAVE_Normalized_BLCorrected'
# data = read_feather(dataName.replace('unprocessed', 'processed_AUTOSAVE_SpikeR_Normalized_BLCorrected')) # CHANGE DATA NAME
# shifts = read_feather('Data-unprocessed_shifts') #unprocessed bc the first file makes this shifts file
# RamanShifts = shifts.iloc[0][0:]
data = read_feather(dataName) 





#%% Body: Outlier detection 100: 0.65 0.66 0.73 0.6 0.56, 80: 0.575, 0.63, 0.66, 0.585, 0.52, 140: 0.8, 0.8, 0.85, 0.75, 0.68

data_dummy = data.copy(deep = True)
if olSwitch:
    print('\noutlier removal in progress..\n')
    # oluif = False # outlier user input
    oluir = False
    for pt in data['Class'].unique():
        if olSwitchRamanBool :#or olSwitchFTIRBool:
            plt.figure()
            ys = data[data['Class'] == pt] #the intensities of this specific plastic type (pt) 
            x = np.linspace(0, ys.shape[1]-1, ys.shape[1]-infoNumCol)
            for pi in range(ys.shape[0]): #going over all samples
                plt.plot(x, ys.iloc[pi,infoNumCol:], linewidth=1, alpha=0.5) #plot every sample
            plt.plot(x, ys.iloc[:,infoNumCol:].mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
                    
            plt.title('Spectra before removing the outliers - ' + pt)
            plt.xlabel( 'Datapoint Index', fontsize = 10)
            plt.ylabel( 'Modified Intensity (a.u.)', fontsize = 10)
        
        
        
        """ OUTLIER DETECTION """
        # # FTIR
        # if olSwitchFTIRBool:
        #     yft = ys.iloc[:,FTIRSeparatorIndex:] #the intensities of this specific plastic type (pt) 
        #     xp = shifts.iloc[0][shiftIndexFTIR:]
        #     [data_z, outlierIndicesFTIR] = outlierDetector(xp, yft,
        #                                           oLevel = oLevel, plotSwitch = False,
        #                                           ramORftir = 'FTIR', plasT = pt)
        #     # original plots
        #     plt.figure()
        #     plt.suptitle( 'Outlier Removal - Threshold Selection - ' + pt)
        #     plt.subplot(1, 2, 1) # row 1, col 2 index 1
        #     for pi in range(yft.shape[0]): #going over all samples
        #         plt.plot(xp, yft.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
        #     plt.plot(xp, yft.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
                    
        #     plt.title('FTIR Spectra ')
        #     plt.xlabel( 'Wavenumber (cm-1)', fontsize = 10)
        #     plt.ylabel( 'Modified Intensity (a.u.)', fontsize = 10)
            
        #     # z-score plots
        #     plt.subplot(1, 2, 2) # row 1, col 2 index 2
        #     yp = data_z #the intensities of this specific plastic type (pt) 
        #     for pi in range(yp.shape[0]): #going over all samples
        #         plt.plot(xp, yp.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
        #     plt.plot(xp, yp.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all z-score plots
        #     # # to increase y axis tick freq:
        #     ax = plt.gca()
        #     # ybeg, yend = ax.get_ylim()
        #     # plt.yticks(np.arange(round(ybeg,0), round(yend,0)+1, 4))        
        #     plt.title('Z-Score Plot')
        #     plt.grid(linewidth=0.4, which='both') 
        #     ax.yaxis.set_minor_locator(AutoMinorLocator()) # minor ticks
        #     plt.grid(linewidth=0.5)
        #     plt.xlabel( 'Wavenumber (cm-1)', fontsize = 10)
        #     plt.ylabel( 'Z-Score (a.u.)', fontsize = 10)  
        #     plt.tight_layout()
                        
        #     # Saving the figure
        #     currdir = os.curdir
        #     os.chdir(mainDirectory + '\\Figures')
        #     figName = pt + " Outlier Removal Plot (FTIR) - Threshold Selection"
        #     plt.savefig(figName, dpi=400, bbox_inches = "tight")
        #     os.chdir(currdir)
        #     plt.show()
            
        #     oluif = input('If outlier removal is required for the presented FTIR spectra input \'y\' please (otherwise, input anything else):\n') # outlier user input
            
        #     if oluif.lower() == 'y': # outlier user input
        #         oluif = True
        #         oLevel = float(input('Input a threshold please:\n'))
        #         [data_z, outlierIndicesFTIR] = outlierDetector(xp, yft,
        #                                               oLevel = oLevel, plotSwitch = olPlotSwitchBool,
        #                                               ramORftir = 'FTIR', plasT = pt)                
        #         data = data.drop(labels = outlierIndicesFTIR, axis = 0, inplace = False) #dropping the outliers out
        #         data = data.reset_index(drop=True) # so that there are no jumps in the sample indices
        #     elif oluif.lower() == '': # outlier user input
        #         oluif = True
        #         oLevel = float(input('Input a threshold please:\n'))
        #         [data_z, outlierIndicesFTIR] = outlierDetector(xp, yft,
        #                                               oLevel = oLevel, plotSwitch = olPlotSwitchBool,
        #                                               ramORftir = 'FTIR', plasT = pt)                
        #         data = data.drop(labels = outlierIndicesFTIR, axis = 0, inplace = False) #dropping the outliers out
        #         data = data.reset_index(drop=True) # so that there are no jumps in the sample indices
        #     else:
        #         oluif = False  # outlier user input       
                
        #     # final plot
        #     if oluif: # outlier user input
        #         plt.figure()
        #         plt.suptitle( 'Outlier Removal (FTIR) - Before-After Comparison - User Confirmation - ' + pt)
        #         plt.subplot(1, 2, 1) # row 1, col 2 index 1
        #         #x as before
        #         for pi in np.arange(ys.shape[0]):
        #             plt.plot(xp, yft.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
        #         plt.plot(xp, yft.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
        #             # if pi == ys.shape[0]-1:
        #             #     plt.plot(x, ys[pi,:], linewidth=1, c='black')
        #             # else:
        #             #     plt.plot(x, ys[pi,:], linewidth=1, alpha=0.8)
        #         plt.title('BEFORE')# - ' + pt)
        #         ymin = ys.iloc[:,FTIRSeparatorIndex:].min().min()
        #         ymax = ys.iloc[:,FTIRSeparatorIndex:].max().max()
        #         plt.ylim(ymin - (0.2 * ymin), ymax + (0.2 * ymax))
        #         plt.xlabel( 'Shifts (cm-1)', fontsize = 10)
        #         plt.ylabel( 'Modified Intensity (a.u.)', fontsize = 10)
                
        #         plt.subplot(1, 2, 2) # row 1, col 2 index 1
        #         yft = data[data['Plastic Type'] == pt].iloc[:,FTIRSeparatorIndex:] # complete data before OL detection
        #         #x as before
        #         for pi in np.arange(yft.shape[0]):
        #             plt.plot(xp, yft.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
        #         plt.plot(xp, yft.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
        #             # if pi == ys.shape[0]-1:
        #             #     plt.plot(x, yra[pi,:], linewidth=1, c='black')
        #             # else:
        #             #     plt.plot(x, ysa[pi,:], linewidth=1, alpha=0.8)
        #         plt.title('AFTER')
        #         plt.ylim(ymin - (0.2 * ymin), ymax + (0.2 * ymax))
        #         ax = plt.gca()
        #         ax.axes.yaxis.set_visible(False)
        #         plt.xlabel( 'Shifts (cm-1)', fontsize = 10)
        #         plt.tight_layout()
                                
        #         # Saving the figure
        #         currdir = os.curdir
        #         os.chdir(mainDirectory + '\\Figures')
        #         figName = pt + " Outlier Removal Plot (FTIR) - Before-After Comparison"
        #         plt.savefig(figName, dpi=400, bbox_inches = "tight")
        #         os.chdir(currdir)
        #         plt.show()
                
        #         #ask for confirmation
        #         ol_conf = input("{}/{} spectra will be deleted (based on FTIR). Do you confirm this result? y: yes, n: undoes the changes.\n".format(len(outlierIndicesFTIR), data[data['Plastic Type'] == pt].shape[0]))
        #         if ol_conf == 'n': 
        #             insIndex = data[data['Plastic Type'] == pt].index[0] # insert should be inserted at this index
        #             data = data.drop(labels = data[data['Plastic Type'] == pt].index,
        #                              axis = 0, inplace = False).reset_index(drop=True)
        #             ins = data_dummy[data_dummy['Plastic Type'] == pt].reset_index(drop=True) # insert from data_dummy
        #             if data_dummy['Plastic Type'].unique()[0] == pt:
        #                 data = concat([ins, data], ignore_index=True)
        #             elif data_dummy['Plastic Type'].unique()[-1] == pt:
        #                 data = concat([data, ins], ignore_index=True)
        #             else:
        #                 data_before_ins = data.iloc[:insIndex,:] #data before insert
        #                 data_after_ins = data.iloc[insIndex:,:].reset_index(drop=True) #data before insert
        #                 data = concat([data_before_ins, ins, data_after_ins], ignore_index=True)
        #             print('The outlier removal (FTIR) changes are undone for {}'.format(pt))
        #         else:
        #             print('\n {}\'s outliers (#{} spectra - based on FTIR\'s shortcoming) are deleted. Indices: {} \n'.format(pt, len(outlierIndicesFTIR), outlierIndicesFTIR))    

        


        """ OUTLIER DETECTION """
        # Raman
        if olSwitchRamanBool:            
            yra = ys.iloc[:,infoNumCol:] #the intensities of this specific plastic type (pt) before OL removal
            xp = shifts.iloc[0][0:]
            [data_z, outlierIndicesRaman] = outlierDetector(xp, yra,
                                                  oLevel = oLevel, plotSwitch = False,
                                                  ramORftir = 'Raman', plasT = pt)
            # original plots
            plt.figure()
            plt.suptitle( 'Outlier Removal (Raman) - Threshold Selection - ' + pt)
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            for pi in range(yra.shape[0]): #going over all samples
                plt.plot(xp, yra.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
            plt.plot(xp, yra.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
                    
            plt.title('Raman Spectra')
            plt.xlabel( 'Raman Shifts (cm-1)',)
            plt.ylabel( 'Modified Intensity (a.u.)')  
            
            
            # z-score plots
            plt.subplot(1, 2, 2) # row 1, col 2 index 2
            yp = data_z #the intensities of this specific plastic type (pt) 
            for pi in range(yp.shape[0]): #going over all samples
                plt.plot(xp, yp.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample's z-score
            plt.plot(xp, yp.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all z-score plots
            # # to increase y axis tick freq:
            ax = plt.gca()
            # ybeg, yend = ax.get_ylim()
            # plt.yticks(np.arange(round(ybeg,0), round(yend,0)+1, 4))        
            plt.title('Z-Score Plot')
            plt.grid(linewidth=0.4, which='both') 
            ax.yaxis.set_minor_locator(AutoMinorLocator()) # minor ticks
            plt.grid(linewidth=0.5)
            plt.xlabel( 'Raman Shifts (cm-1)')
            plt.ylabel( 'Z-Score (a.u.)' )
            plt.tight_layout()
                        
            # Saving the figure
            currdir = os.curdir
            os.chdir(mainDirectory + '\\Figures')
            figName = pt + " Outlier Removal Plot (Raman) - Threshold Selection"
            plt.savefig(figName, dpi=400, bbox_inches = "tight")
            os.chdir(currdir)
            plt.show()
            
            oluir = input('If outlier removal is required for Raman spectra input y please (otherwise, input anything else):\n') # outlier user input
            
            if oluir.lower() == 'y': # outlier user input
                oluir = True 
                oLevel = float(input('Input a threshold please:\n'))
                [data_z, outlierIndicesRaman] = outlierDetector(shifts.iloc[0][0:],
                                                      ys.iloc[:,infoNumCol:],
                                                      oLevel = oLevel, plotSwitch = olPlotSwitchBool,
                                                      ramORftir = 'Raman', plasT = pt)
                data = data.drop(labels = outlierIndicesRaman, axis = 0, inplace = False) #dropping the outliers out
                data = data.reset_index(drop=True) # so that there are no jumps in the sample indices
            else:
                oluir = False # outlier user input
            
            
            # final plot
            if oluir: # outlier user input
                plt.figure()
                plt.suptitle( 'Outlier Removal (Raman) - Before-After Comparison - User Confirmation - ' + pt)
                plt.subplot(1, 2, 1) # row 1, col 2 index 1
                #x as before
                for pi in np.arange(ys.shape[0]):
                    plt.plot(xp, yra.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
                plt.plot(xp, yra.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
                    # if pi == ys.shape[0]-1:
                    #     plt.plot(x, ys[pi,:], linewidth=1, c='black')
                    # else:
                    #     plt.plot(x, ys[pi,:], linewidth=1, alpha=0.8)
                plt.title('BEFORE')# - ' + pt)
                ymin = yra.min().min()
                ymax = yra.max().max()
                plt.ylim(ymin - (0.2 * ymax), ymax + (0.2 * ymax))
                plt.xlabel( 'Raman Shifts (cm-1)')
                plt.ylabel( 'Modified Intensity (a.u.)')
                
                plt.subplot(1, 2, 2) # row 1, col 2 index 1
                yra = data[data['Class'] == pt].iloc[:,infoNumCol:] # this copy of data does not have the outliers
                #x as before
                for pi in np.arange(yra.shape[0]):
                    plt.plot(xp, yra.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
                plt.plot(xp, yra.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
                    # if pi == ys.shape[0]-1:
                    #     plt.plot(x, yra[pi,:], linewidth=1, c='black')
                    # else:
                    #     plt.plot(x, ysa[pi,:], linewidth=1, alpha=0.8)
                plt.title('AFTER')
                plt.ylim(ymin - (0.2 * ymax), ymax + (0.2 * ymax))
                ax = plt.gca()
                ax.axes.yaxis.set_visible(False)
                plt.xlabel( 'Raman Shifts (cm-1)')
                plt.tight_layout()
                        
                # Saving the figure
                currdir = os.curdir
                os.chdir(mainDirectory + '\\Figures')
                figName = pt + " Outlier Removal Plot (Raman) - Before-After Comparison"
                plt.savefig(figName, dpi=400, bbox_inches = "tight")
                os.chdir(currdir)
                plt.show()
                
                #ask for confirmation
                ol_conf = input("{}/{} spectra will be deleted (based on Raman). Do you confirm this result? y: yes, n: undoes the changes.\n".format(len(outlierIndicesRaman), data_dummy[data_dummy['Class'] == pt].shape[0]))
                if ol_conf == 'n': 
                    insIndex = data[data['Class'] == pt].index[0] # insert should be inserted at this index
                    data = data.drop(labels = data[data['Class'] == pt].index,
                                      axis = 0, inplace = False).reset_index(drop=True)
                    ins = data_dummy[data_dummy['Class'] == pt].reset_index(drop=True) # insert from data_dummy
                    if data_dummy['Class'].unique()[0] == pt:
                        data = concat([ins, data], ignore_index=True)
                    elif data_dummy['Class'].unique()[-1] == pt:
                        data = concat([data, ins], ignore_index=True)
                    else:
                        data_before_ins = data.iloc[:insIndex,:] #data before insert
                        data_after_ins = data.iloc[insIndex:,:].reset_index(drop=True) #data before insert
                        data = concat([data_before_ins, ins, data_after_ins], ignore_index=True)
                    print('The outlier removal (Raman) changes are undone for {}'.format(pt))
                else:
                    print('\n {}\'s outliers (#{} spectra - based on Raman\'s shortcoming) are deleted. Indices: {} \n'.format(pt, len(outlierIndicesRaman), outlierIndicesRaman))





#%% Data Export - mid point save

print('\n\n Exporting the processed data before spike removal.')

os.chdir(mainDirectory)
dataName = dataName.replace('processed_AUTOSAVE_Normalized_BLCorrected', 'processed_AUTOSAVE_OutL_Normalized_BLCorrected')
data = data.reset_index(drop=True)
#data.to_excel("{}.xlsx".format(dataName)) 
data.to_feather("{}".format(dataName)) 



#%% Loading Intermediate Save
# os.chdir(mainDirectory)
# dataName = 'Data-processed_AUTOSAVE_OutL_Normalized_BLCorrected'
# #data = read_feather(dataName.replace('unprocessed', 'processed_AUTOSAVE_SpikeR_Normalized_BLCorrected')) # CHANGE DATA NAME
# data = read_feather(dataName) # CHANGE DATA NAME


    
    
#%% SPIKE REMOVAL    

#spike removal should be separate so the spectra's baselines and intensities are already corrected and normalized
if spikeSwitchRamanBool:    
    print('\nRaman spectra\'s spike removal in progress..\n')
    pt = data['Class'][0]
    userconfSwitch = input('Do you want user confirmation for the spike removal: y (yes): \n')
    for i in tqdm(range(datalimMIN, data.shape[0])):
        # FTIRSpec = data.iloc[i][FTIRSeparatorIndex:]
    
        RamanSpec = data.iloc[i][infoNumCol:]
        #Ramanshifts are assigned priorly
        #RamanShifts = shifts.iloc[0][0:shiftIndexFTIR]
        
    
        """ Raman SPIKE removal """
            
        if data['Class'][i] != pt or i == 0: #if the plastic type changes, user should change the threshold
            pt = data['Class'][i]
            ys = data[data['Class'] == pt] #the fused intensities of this specific plastic type (pt)
            ys_dummy = ys.copy(deep = True) # for making a comparative plot before after spr
            yra = ys.iloc[:,infoNumCol:] #the Raman intensities of this specific plastic type (pt) 
            xp = shifts.iloc[0][0:] # Raman shifts
            if spr_method == 'wh' or spr_method == 'wh_collective': #options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
                [data_z, outlierIndicesRaman] = outlierDetector(xp, yra,
                                                      oLevel = oLevel, plotSwitch = False,
                                                      ramORftir = 'Raman', plasT = pt)
            elif spr_method == 'mr': #options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
                # here we will produce the deltas based on which the removal works
                yra_delta = yra.copy(deep = True)
                thresholds = []
                for jk in np.arange(yra.shape[0]):
                    intensities = yra.iloc[jk,:]
                    intensities_diff = np.diff(intensities) # formula 2 in the paper - D
                    N = len(intensities_diff) # length of spectrum
                    
                    # calculating Allan deviation = ADEV
                    acum = 0 # for the summation
                    for ci in np.arange(N-1): # summation
                        acum += (intensities_diff[ci+1] - intensities_diff[ci])**2
                    adev = np.sqrt(acum / (2*(N - 2))) # formula 6 in the paper
                    threshold = RamanThreshold * adev # as in the case of Z-score, we need a threshold
                    thresholds.append(threshold)
                    
                    [intensities_diff_dwt, _] = pywt.dwt(intensities_diff, wname) #wname: wavelet name, returns approximation coefficients vector and detail coefficients vector
                    intensities_diff_filtered = pywt.idwt(intensities_diff_dwt, None, wname) # filtered D, we are losing the detail coefficients vector by passing none instead
                    intensities_diff_filtered = intensities_diff_filtered[:N]
                    deltas = intensities_diff - intensities_diff_filtered #residuals - formula 7 of the paper
                    yra_delta.iloc[jk,:] = concat([DataFrame([0]), DataFrame(deltas)], axis = 0).values.reshape(yra_delta.iloc[jk,:].shape[0],)

            
            
            
            plt.figure()
            # original plots
            plt.suptitle( 'Spike Removal - Raman - Threshold Selection - ' + pt)
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            for pi in range(yra.shape[0]): #going over all samples
                plt.plot(xp, yra.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
            plt.plot(xp, yra.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
            # ymin = yra.min().min()
            # ymax = yra.max().max()
            # plt.ylim(ymin - (0.2 * ymin), ymax + (0.2 * ymax))        
            plt.title('Raman Spectra')
            plt.xlabel( 'Raman Shifts (cm-1)')
            plt.ylabel( 'Modified Intensity (a.u.)')
            
            if spr_method == 'wh' or spr_method == 'wh_collective': #options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
                # z-score plots
                plt.subplot(1, 2, 2) # row 1, col 2 index 1
                yp = data_z #the intensities of this specific plastic type (pt) 
                for pi in range(yp.shape[0]): #going over all samples
                    plt.plot(xp, yp.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample    
                plt.plot(xp, yp.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
                # # to increase y axis tick freq:
                ax = plt.gca()
                # ybeg, yend = ax.get_ylim()
                # plt.yticks(np.arange(round(ybeg,0), round(yend,0)+1, 4))
                plt.title('Z-score Plot')
                plt.grid(linewidth=0.4, which='both') 
                ax.yaxis.set_minor_locator(AutoMinorLocator()) # minor ticks
                plt.xlabel( 'Raman Shifts (cm-1)')
                plt.ylabel( 'Z-Score (a.u.)') 
                plt.tight_layout()
                            
                # Saving the figure
                currdir = os.curdir
                os.chdir(mainDirectory + '\\Figures')
                figName = pt + " Spike Removal Plot - Choice of Threshold"
                plt.savefig(figName, dpi=400, bbox_inches = "tight")
                os.chdir(currdir)
                plt.show()
            
            elif spr_method == 'mr': #options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
                # delta D plots
                plt.subplot(1, 2, 2) # row 1, col 2 index 1
                yp = yra_delta #the intensities of this specific plastic type (pt) 
                for pi in range(yp.shape[0]): #going over all samples
                    plt.plot(xp, yp.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample    
                plt.plot(xp, yp.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
                plt.axhline(np.average(thresholds), ls = '--', c = 'grey', alpha = 0.5)
                plt.title('Raman Delta D spectra - ' + pt)
                plt.grid(linewidth=0.5)
                plt.xlabel( 'Raman Shifts (cm-1)')
                plt.ylabel( 'Delta D (a.u.)')  
                plt.tight_layout()
                                
                # Saving the figure
                currdir = os.curdir
                os.chdir(mainDirectory + '\\Figures')
                figName = pt + " Spike Removal Plot - Choice of Threshold"
                plt.savefig(figName, dpi=400, bbox_inches = "tight")
                os.chdir(currdir)
                plt.show()
            
            RamanThreshold = float(input('Please note the graph and choose an appropriate threshold for spike removal ({}) - current threshold is {}:\n'.format(spr_method, RamanThreshold)))
            
            
            
            
        #spike removal by averaging the detected spike or interpolation   
        if spr_method == 'wh' or spr_method == 'wh_collective': #options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
            RamanSpec = spikeRemoval(RamanShifts, RamanSpec, ramORftir = 'Raman', data_z = data_z,
                                      scoreType = 'Whitaker-Hayes',
                                      m=3, threshold = RamanThreshold,
                                      spr_method = spr_method, 
                                      userconfSwitch = userconfSwitch) # options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
        elif spr_method == 'mr': #options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla
            RamanSpec = spikeRemoval(RamanShifts, RamanSpec, ramORftir = 'Raman',
                                      scoreType = 'Whitaker-Hayes',
                                      m=3, threshold = RamanThreshold,
                                      spr_method = spr_method, 
                                      userconfSwitch = userconfSwitch) # options: 'wh':Whitaker-Hayes, 'wh_collective':Whitaker-Hayes Collective, 'mr': Maury-Revilla

        #updated_Row = np.concatenate([RamanSpec,
        #                              FTIRSpec]).transpose()
        updated_Row = RamanSpec.transpose().reset_index(drop=True)

        
        data.iloc[i, infoNumCol:] = updated_Row
        
        
        
        
        #making comparative plots when the pt samples are processed
        if i == data[data['Class'] == pt].index[-1]:
            yra = ys_dummy.iloc[:,infoNumCol:] #the Raman intensities of this specific plastic type (pt) 
            # xp = shifts.iloc[0][0:shiftIndexFTIR] # Raman shifts - already assigned

            plt.figure()
            plt.suptitle( 'Spike Removal - Raman - Final Comparison Before-After Plot - ' + pt)
            # original plots
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            for pi in range(yra.shape[0]): #going over all samples
                plt.plot(xp, yra.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample
            plt.plot(xp, yra.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
            # ymin = yra.min().min()
            # ymax = yra.max().max()
            # plt.ylim(ymin - (0.2 * ymin), ymax + (0.2 * ymax))        
            plt.title('BEFORE')
            plt.xlabel( 'Raman Shifts (cm-1)')
            plt.ylabel( 'Modified Intensity (a.u.)')
            
            # z-score plots
            ys = data[data['Class'] == pt] #the fused intensities of this specific plastic type (pt) #the intensities of this specific plastic type (pt) 
            yp = ys.iloc[:,infoNumCol:] #the Raman intensities of this specific plastic type (pt) 
            plt.subplot(1, 2, 2) # row 1, col 2 index 1
            for pi in range(yp.shape[0]): #going over all samples
                plt.plot(xp, yp.iloc[pi,:], linewidth=1, alpha=0.5) #plot every sample    
            plt.plot(xp, yp.mean(axis=0), linewidth=0.5, c='black') #plot the mean of all samples
            plt.title('AFTER')
            plt.xlabel( 'Raman Shifts (cm-1)')
            plt.ylabel( 'Modified Intensity (a.u.)')  
            plt.tight_layout()
                        
            # Saving the figure
            currdir = os.curdir
            os.chdir(mainDirectory + '\\Figures')
            figName = pt + " Spike Removal Plot - Before-After Comparison"
            plt.savefig(figName, dpi=400, bbox_inches = "tight")
            os.chdir(currdir)
            plt.show()
    



print('\nData modifications are done.\n')



#%% Test Sample

# intensity = np.array(data[320:321].values[0][3:])
# plt.figure()
# plt.plot(np.linspace(0, len(intensity)-1, len(intensity)), intensity)
# plt.vlines(shiftIndexFTIR, min(intensity), abs(min(intensity)), colors='red', linestyles='--')
# plt.xlabel( 'Datapoint Index', fontsize = 10)
# plt.ylabel( 'Modified Intensity (a.u.)', fontsize = 10)

# intensity = np.array(data[319:320].values[0][3:])
# plt.figure()
# plt.plot(np.linspace(0, len(intensity)-1, len(intensity)), intensity)
# plt.vlines(shiftIndexFTIR, min(intensity), abs(min(intensity)), colors='red', linestyles='--')
# plt.xlabel( 'Datapoint Index', fontsize = 10)
# plt.ylabel( 'Modified Intensity (a.u.)', fontsize = 10)

    



#%% Data Export - mid point save
print('\n\n Exporting the processed data before smoothing.')
os.chdir(mainDirectory)
#dataName = dataName.replace('unprocessed', 'processed_AUTOSAVE_OutL_SpikeR_Normalized_BLCorrected')
dataName = dataName.replace('processed_AUTOSAVE_OutL_Normalized_BLCorrected', 'processed_AUTOSAVE_SpikeR_OutL_Normalized_BLCorrected')
data = data.reset_index(drop=True)
#data.to_excel("{}.xlsx".format(dataName))
data.to_feather("{}".format(dataName))
#%% Loading Intermediate Save
# os.chdir(mainDirectory)
# dataName = 'Data-processed_AUTOSAVE_OutL_SpikeR_Normalized_BLCorrected'
# data = read_feather(dataName)#.replace('unprocessed', 'processed_AUTOSAVE_SpikeR_OutL_Normalized_BLCorrected')) # CHANGE DATA NAME




#%% Smoothing
#data = read_feather(dataName.replace('unprocessed', 'processed_SR_Normalized_BLCorrected')) # CHANGE DATA NAME
print('\nSavitzky-Golay smoothing in progress..\n')
if SGFilterSwitch == 1:
    filterName = 'Savitzky-Golay'
    for i in tqdm(range(datalimMIN, data.shape[0])):  
        pt = data['Class'][i]
        #resultsMatSG = np.zeros((len(all_Data[0,:]), qFactors)) #Intensity, RShift, FWHM, SNR
        if i == 0:
            RamanShifts = shifts.iloc[0][0:]
            sgWin, sgOrder = SGOptimizer(data.iloc[i,infoNumCol:],
                                         RamanShifts.values,
                                         sgWinRange, sgOrderRange)
        dataVecSG = savitzky_golay(data.iloc[i,infoNumCol:],
                                   window_size=sgWin, order=sgOrder) #with backrgound
        
        # Visualization
        if SGPlotSwitch == 1:
            os.chdir(mainDirectory)
            if not os.path.exists('Figures'):
                os.makedirs('Figures')
            os.chdir(mainDirectory + '\\Figures')
            if not os.path.exists(filterName + ' - Spectral Smoothing'):
                os.makedirs(filterName + ' - Spectral Smoothing')
            os.chdir(mainDirectory + '\\Figures' + '\\' + filterName + ' - Spectral Smoothing')
            
            if data[data['Class'] == pt].index[0] < i < data[data['Class'] == pt].index[0] + 10:
                if not os.path.exists(pt):
                    os.makedirs(pt)
                os.chdir(mainDirectory + '\\Figures' + '\\' + filterName + ' - Spectral Smoothing\\' +pt)
                            
                plt.figure()
                plt.suptitle( 'Spectral Smoothing - ' + filterName + ' - ' +  pt)
                plt.subplot(2, 1, 1) # row 1, col 1 index 1
                xp = shifts.values.reshape(data.iloc[i,infoNumCol:].shape)
                plt.plot(xp, data.iloc[i,infoNumCol:]) #plot unsmoothed spectrum
    
                plt.title('BEFORE')# - ' + pt)
                #plt.xlabel( 'Raman Shifts (cm-1)')
                plt.ylabel( 'Modified Intensity (a.u.)')
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                
                plt.subplot(2, 1, 2) # row 2, col 1 index 2
                #x as before
                plt.plot(xp, dataVecSG) #plot smoothed spectrum
                plt.title('AFTER')
                plt.xlabel( 'Raman Shifts (cm-1)')
                plt.ylabel( 'Modified Intensity (a.u.)')
                plt.tight_layout()
                                
                # Saving the figure
                figName = pt + " Spectral Smoothing Plot - Before-After Comparison - Sample {}".format(i)
                plt.savefig(figName, dpi=400, bbox_inches = "tight")
                os.chdir(mainDirectory)
                plt.close()
        
        #Final Assignment of the smoothed data
        data.iloc[i,infoNumCol:] = dataVecSG




#%% Data Export
print('\n\n Exporting the processed data.')

os.chdir(mainDirectory)
dataName = 'Data-preprocessed'
data = data.reset_index(drop=True)
#data.to_excel("{}.xlsx".format(dataName)) 
data.to_feather("{}".format(dataName)) 





# excelName = 'dataNewCheckNaN.xlsx'
# excelFile = ExcelWriter(excelName, engine='xlsxwriter') #pandas
# data.to_excel(excelFile, sheet_name='sheet1')
# excelFile.save()  




#%% Test Sample
#os.chdir(mainDirectory)
#data = read_feather(dataName.replace('unprocessed', 'preprocessed')) # CHANGE DATA NAME
# os.chdir(mainDirectory + '\\Figures')
# if not os.path.exists('Random Example Spectra'):
#     os.makedirs('Random Example Spectra')
# os.chdir(mainDirectory + '\\Figures\\Random Example Spectra')
# for pt in unique(data['Class']):
#     if not os.path.exists(pt):
#         os.makedirs(pt)
#     os.chdir(mainDirectory + '\\Figures\\Random Example Spectra\\'+pt)
#     xintensity = data.iloc[random.sample(list(data[data['Class'] == pt].index.values), k=10), :].iloc[:,infoNumCol:]
#     for intInd in np.arange(xintensity.shape[0]):
#         intensity = xintensity.iloc[intInd,:].values
#         plt.figure()
#         plt.plot(shifts.values.reshape((len(intensity),)), intensity) #np.arange(len(intensity))
#         #plt.vlines(shiftIndexFTIR, min(intensity), abs(min(intensity)), colors='red', linestyles='--')
#         plt.xlabel( 'Raman Shift (cm-1)')
#         plt.ylabel( 'Modified Intensity (a.u.)')
#         plt.title('Example Spectrum - {} - Sample:{}'.format(pt, xintensity.iloc[intInd,:].name))
#         # Saving the figure
#         figName = 'Example Spectrum - {} - Sample {}'.format(pt, xintensity.iloc[intInd,:].name) + '.jpg'
#         plt.savefig(figName, dpi=400, bbox_inches = "tight")    
#         plt.close
#     os.chdir(mainDirectory + '\\Figures\\Random Example Spectra')
    
    
#%% Raman and FTIR separately
#FTIRSpecs = data.drop(data.iloc[:,infoNumCol:FTIRSeparatorIndex].columns, axis=1).reset_index(drop=True)
# RamanSpecs = data#.iloc[:, :FTIRSeparatorIndex]
# os.chdir(mainDirectory)
# #data.to_excel("{}.xlsx".format(dataName))
# RamanSpecs.to_feather("{}".format(dataName.replace('unprocessed', 'preprocessed_Raman')))
#data.to_excel("{}.xlsx".format(dataName))
#FTIRSpecs.to_feather("{}".format(dataName.replace('unprocessed', 'preprocessed_FTIR')))















