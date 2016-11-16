# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 2016

@author: Marc Stefanon
"""
# Library for challenge 28 (ANAP)
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import glob, os
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

########################################################################
# Loading data
def loadTrain():
    FilePath = "/homedata/mstefano/ANAP/"
    FileLoc = FilePath + 'DataTrain.npz'
    Train = np.load(FileLoc)
    clef = Train.files
    Target = Train[clef[0]]
    Age = Train[clef[1]]
    SejourALD = Train[clef[2]]
    Origin = Train[clef[3]]
    Sejourtot = Train[clef[4]]
    Finess = Train[clef[5]]
    Year = Train[clef[6]]
    Speciality = Train[clef[7]]
    return (Target,Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality)

def loadTest():
    FilePath = "/homedata/mstefano/ANAP/"
    FileLoc = FilePath + 'DataTest.npz'
    Test = np.load(FileLoc)
    clef = Test.files
    Origin = Test[clef[0]]
    Sejourtot = Test[clef[1]]
    Finess = Test[clef[2]]
    SejourALD = Test[clef[3]]
    Age = Test[clef[4]]
    Year = Test[clef[5]]
    Speciality = Test[clef[6]]
    return (Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality)

########################################################################
# Retrieving Hospital department from Hospital Id
def HospitalCounty(Finess):
    FiOrigin = []
    for x in np.arange(0,len(Finess)):
        if  len(str(Finess[x]))  == 8: 
            FiOrigin.append(float(str(Finess[x])[0]))
        if  len(str(Finess[x]))  == 9: 
            if  float(str(Finess[x])[0:2])  == 97:                
                FiOrigin.append(float(str(Finess[x])[0:2] + str(Finess[x])[3]))
            else:
                FiOrigin.append(float(str(Finess[x])[0:2]))
        if  len(str(Finess[x]))  == 10: 
            FiOrigin.append(float(str(Finess[x])[0:3]))
    return np.asarray(FiOrigin)

# table correspondance 100x100 | cat. 1 = meme departement; cat 2 = meme région; cat 3 = autre;
# table correspondance 100x100 | cat. 1 = meme departement; cat 2 = departement voisin; cat 3 = autre;

########################################################################
# Writing Result
def WriteRes(Res):
    FilePath = "/homedata/mstefano/ANAP/"
    FileLoc = FilePath + 'Resultat_Stefanon.csv'; 
    print("Directory Output:  \n" + FileLoc)
    FileId  = open(FileLoc,"wt")
    data = csv.writer(FileId, delimiter=';')
    data.writerow(["id","cible"])
    indice  = np.arange(0,len(Res),dtype=int)
    dicti = np.vstack((indice,Res)).transpose()
    for row in dicti:
        data.writerow(row)
    FileId.close()
    return
#:%s/.0;/;/g


########################################################################
# Metric definition
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nrmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean()) / (np.max(targets) - np.min(targets))

########################################################################
# White noise
def Whitenoise(num_samples):
    mean = 0; std = 1;
    WhiteNoise = []
    for x in np.arange(0,1000):
            samplesUn = np.random.normal(mean, std, size=num_samples)
            samplesDeux = np.random.normal(mean, std, size=num_samples)
            WhiteNoise.append(np.corrcoef(samplesUn,samplesDeux)[1,0])
    return np.asarray(WhiteNoise)


########################################################################
# Data Aggregation for graphics, per dx 
def aggregate(dx,Target,Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality,ratioALD):
    NTarget = np.zeros((dx,3)); NAge = np.zeros((dx,3)); 
    NOrigin = np.zeros((dx,3)); NSejourtot = np.zeros((dx,3)); 
    NFiness = np.zeros((dx,3)); NYear = np.zeros((dx,3)); 
    NSpeciality = np.zeros((dx,3)); NSejourALD = np.zeros((dx,3)); 
    NratioALD = np.zeros((dx,3)); 
    for x in np.arange(0,100):
        NTarget[x,0] = Target[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NTarget[x,1] = len(Target[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NTarget[x,2] = Target[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NAge[x,0] = Age[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NAge[x,1] = len(Age[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NAge[x,2] = Age[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NOrigin[x,0] = Origin[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NOrigin[x,1] = len(Origin[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NOrigin[x,2] = Origin[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NSejourtot[x,0]  = Sejourtot[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NSejourtot[x,1]  = len(Sejourtot[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NSejourtot[x,2]  = Sejourtot[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NFiness[x,0] = Finess[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NFiness[x,1] = len(Finess[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NFiness[x,2] = Finess[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NYear[x,0] = Year[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NYear[x,1] = len(Year[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NYear[x,2] = Year[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NSpeciality[x,0] = Speciality[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NSpeciality[x,1] = len(Speciality[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NSpeciality[x,2] = Speciality[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NSejourALD[x,0] = SejourALD[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NSejourALD[x,1] = len(SejourALD[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NSejourALD[x,2] = SejourALD[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
        NratioALD[x,0] = ratioALD[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NratioALD[x,1] = len(ratioALD[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NratioALD[x,2] = ratioALD[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
    return (NTarget,NAge,NSejourALD,NSejourtot,NFiness,NOrigin,NYear,NSpeciality,NratioALD)

# Data Aggregation for graphics, per dx 
def aggregateRes(dx,Res,ratioALD):
    NRes = np.zeros((dx,3)); 
    for x in np.arange(0,100):
        NRes[x,0] = Res[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].mean()
        NRes[x,1] = len(Res[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)]) / float(len(ratioALD))
        NRes[x,2] = Res[ np.logical_and(ratioALD > x/dx,ratioALD <= (x+1)/dx)].std()
    return (NRes)


########################################################################
def f(x):
    return np.float(x)


