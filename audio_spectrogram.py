#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import pandas
from os import path
from matplotlib import cm
import math
import scipy

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec,  untillfreq, fromfreq, sr=44100):
    timebins, freqbins = np.shape(spec)
    #print (freqbins)

    scale = np.logspace(0, np.log10(freqbins), freqbins, base=10)

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        bool1=freqbins-2>i and (scale[i+1]-scale[i])<1
        if  bool1:
            newspec[:,i] = spec[:,int(np.round(scale[i]))]
        elif i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        bool1=freqbins-2>i and (scale[i+1]-scale[i])<1
        if bool1:
            freqs += [(allfreqs[int(np.round(scale[i]))])]
        elif i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i])])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]
##    print (len(freqs), newspec.shape)
    index_max=next(idx for idx, value in enumerate(freqs) if value > untillfreq)
    index_min=next(idx for idx, value in enumerate(freqs) if value > fromfreq)
    return newspec[:,index_min:index_max], freqs[index_min:index_max]

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**15, plotpath=None, colormap="PuRd"):
    samplerate, samples = wav.read(audiopath)
    mat = scipy.io.loadmat(audiopath[0:-4] + '_calibration.mat')
    samples = samples/float(2**16-1)*mat['Kal']
    s = stft(samples, binsize, overlapFac=0.82)
    sshow, freq = logscale_spec(s, untillfreq=9000, fromfreq=8.5, sr=samplerate)#added untillfreq for "ylim"
    ims = 10.*np.log10(np.abs(sshow)/(2*10**-5)) # amplitude to decibel

##    Leq = 10.*np.log10( np.mean(samples**2)/((2*10**-5)**2) );

    timebins, freqbins = np.shape(ims)

    #print("timebins: ", timebins)
    #print("freqbins: ", freqbins)

    yticklabels=[10, 25, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 8000]
    ytickpixel=[]
    for ik in range(0,len(yticklabels)):
        ytickpixel.append(min(range(len(freq)), key=lambda i: abs(freq[i]-yticklabels[ik])))

    fig, ax = plt.subplots(figsize=(11.69,8.27))
##    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="hanning")
    cbar=plt.colorbar()
     
    for font_objects in cbar.ax.yaxis.get_ticklabels():
        font_objects.set_size(15)

    cbar.set_label('[dB]',fontsize = 15)

    plt.xlabel("Time (s)",fontsize=15)
    plt.ylabel("Frequency (Hz)",fontsize=15)
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    plt.clim(40,80)
    ax.set_yticklabels(yticklabels)
    plt.yticks(ytickpixel)
    xlocs = np.linspace(0,timebins,13)
    plt.xticks(xlocs, [ '%1.0f' % i for i in np.linspace(0,np.floor(np.size(samples)/samplerate),13)])
    plt.xticks(fontsize=15, rotation=0)
    plt.yticks(fontsize=15, rotation=0)
    

    if plotpath:
        plt.savefig(plotpath, dpi=150, orientation= 'landscape', papertype='a4')
    else:
        plt.show()

    plt.clf()
    plt.close()

    return ims
