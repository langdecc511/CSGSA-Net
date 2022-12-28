# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:13:32 2021
@author: wangxu
"""
import matplotlib; 
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from sklearn import preprocessing
from scipy import signal

import pyedflib
from scipy.signal import butter, filtfilt

from sklearn.model_selection import train_test_split

select_index = 0

def NormalizeData1(v):
    v_min = v.min(axis=1).reshape((v.shape[0],1))
    v_max = v.max(axis=1).reshape((v.shape[0],1))
    return (v - v_min) / (v_max-v_min)



def NormalizeData(v):
    return (v - v.mean(axis=1).reshape((v.shape[0],1))) / (v.max(axis=1).reshape((v.shape[0],1)) + 2e-12) 


# wangxu add starting
def get_filelist(path): 
    Filelist = []
    Filelist_final = []
    Filelist_test = []
    for home, dirs, files in os.walk(path):
        for filename in files: 
            spl = filename.split('.')
            filen = os.path.join(home, spl[0])
            if filen not in Filelist:
                Filelist.append(filen)         
                
    for string in Filelist: 
        string1 = string.rsplit('/', 1)
        string2 = string1[-1].rsplit('_',1)
        string_file = string1[0] + '/' + string2[0]
        
        if os.path.exists(string_file + '_fecg1.dat') or os.path.exists(string_file + '_fecg2.dat') :
            Filelist_final.append(string_file)
        
    test = pd.DataFrame(data=Filelist_final) 
    Filelist_final = test.drop_duplicates()
    
    return Filelist_final




def windowingSig(sig1, sig2, windowSize=128):
        signalLen = sig2.shape[1]
        signalsWindow1 = [sig1[:, int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize +1, windowSize)]
        signalsWindow2 = [sig2[:, int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize +1, windowSize)]

        return signalsWindow1, signalsWindow2


# wangxu add ending


txt = []
class FECGDataset(Dataset):
    def __init__(self, data_path="./ADFECGDB/",train=True):
        super(FECGDataset, self).__init__()
        self.fileNames= ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]
        
        ecgWindows, fecgWindows = self.prepareData(delay=5,train=train)
        self.X_train, _, self.Y_train, _ = self.trainTestSplit(ecgWindows, fecgWindows, len(ecgWindows)-1)
        
        
    def readData(self, sigNum, path="./ADFECGDB/"):
        file_name = path + self.fileNames[sigNum]
        f = pyedflib.EdfReader(file_name)
        n = f.signals_in_file
        # signal_labels = f.getSignalLabels()
        abdECG = np.zeros((n - 1, f.getNSamples()[0]))
        fetalECG = np.zeros((1, f.getNSamples()[0]))
        fetalECG[0, :] = f.readSignal(0)
        fetalECG[0, :] = scale(self.butter_bandpass_filter(fetalECG, 1, 100, 1000), axis=1)
        for i in np.arange(1, n):
            abdECG[i - 1, :] = f.readSignal(i)
        abdECG = scale(self.butter_bandpass_filter(abdECG, 1, 100, 1000), axis=1)


        abdECG = signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
        fetalECG = signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1)
        return abdECG, fetalECG
    
    
    def prepareData(self, delay=5,train=True):
        ecgAll, fecgAll=None,None
        if train:
            ecgAll, fecg = self.readData(1)
            ecgAll = ecgAll[range(2,3), :]
            ecgAll = NormalizeData(ecgAll)
            delayNum = ecgAll.shape[0]
            fecgAll = self.createDelayRepetition(fecg, delayNum, delay)
            fecgAll = NormalizeData(fecgAll)
            
            
            for i in range(2, 5):
                ecg, fecg = self.readData(i)
                ecg = ecg[range(2,3), :]
                ecg = NormalizeData(ecg)
                fecgDelayed = self.createDelayRepetition(fecg, delayNum, delay)
                fecgDelayed = NormalizeData(fecgDelayed)
                ecgAll = np.append(ecgAll, ecg, axis=1)
                fecgAll = np.append(fecgAll, fecgDelayed, axis=1)
        else:
            ecgAll, fecg = self.readData(select_index)
            ecgAll = ecgAll[range(2,3), :]
            ecgAll = NormalizeData(ecgAll)
            delayNum = ecgAll.shape[0]
            fecgAll = self.createDelayRepetition(fecg, delayNum, delay)
            fecgAll = NormalizeData(fecgAll)
            

        ecgWindows, fecgWindows = windowingSig(ecgAll, fecgAll, windowSize=128)
        return ecgWindows, fecgWindows

    def trainTestSplit(self, sig, label, trainPercent, shuffle=False):
        X_train, X_test, y_train, y_test = train_test_split(sig, label, train_size=trainPercent, shuffle=shuffle)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test
    
    def createDelayRepetition(self, signal, numberDelay=4, delay=10):
        signal = np.repeat(signal, numberDelay, axis=0)
        for row in range(1, signal.shape[0]):
            signal[row, :] = np.roll(signal[row, :], shift=delay * row)
        return signal

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=1):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
        return y
    
    
    
    
    
    
    
    
    
    

    def __getitem__(self, index):        
        dataset_x = self.X_train[index,:,:]
        dataset_y = self.Y_train[index,:,:]
        return dataset_x, dataset_y
        # return dataset_x, dataset_y, data_xx_idx

    def __len__(self):
        return self.X_train.shape[0]
    

if __name__ == '__main__':
    d = FECGDataset(config.train_data)
    print(d[0])
