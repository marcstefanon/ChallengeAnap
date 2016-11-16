# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2 2016

@author: Marc Stefanon
"""
# Sources
# http://www.commentcamarche.net/faq/2382-python-lire-et-ecrire-des-fichiers-csv
########################################################################
# File preparation : Removing commas ","
# :%s/,//g
########################################################################
import csv
import glob, os
import numpy as np

########################################################################
# Open csv file with training data
FilePath = "/homedata/mstefano/ANAP/"
csv_files = [f for f in os.listdir(FilePath) if f.endswith('.csv')]

FileLoc = FilePath + csv_files[0]; print(FileLoc)
FileId  = open(FileLoc,"rt")
data = csv.reader(FileId)
numline = len(FileId.readlines()); 

# Skip first line
FileId.seek(0); next(data); 

# Initialize variables
Finess = []; Origin = []; Speciality = [];
Age = []; SejourALD = []; Sejourtot = [];
Year = []; Target = []

# Read and store informations
i = 0; cpt = round(numline / 100);
for row in data:
   i += 1
   if ( i % (2*cpt) == 0):
         print( round((100*i / numline),2) , ' %')
   temp = row[0].split(';');
   Finess.append(temp[0]); Origin.append(temp[2][0:3]); Speciality.append(temp[3][1:3]);
   Age.append(temp[4]); SejourALD.append(temp[5]); Sejourtot.append(temp[6]);
   Year.append(temp[7]); Target.append(temp[8])

FileId.close()

# Convert from list of strings to numpy array
Age  = [0 if x=='<=75 ans' else 1 for x in Age]; Age = np.asarray(Age);
Speciality = list(map(int, Speciality)); Speciality = np.asarray(Speciality);
SejourALD = list(map(int, SejourALD)); SejourALD = np.asarray(SejourALD);
Sejourtot = list(map(int, Sejourtot)); Sejourtot = np.asarray(Sejourtot);
Year = list(map(int, Year)); Year = np.asarray(Year);
Target = list(map(float, Target)); Target = np.asarray(Target);

# Post processing department numbering
Origin  = ['0' if x=='Inc' else x for x in Origin]; 
Origin  = [x[0:2] if x[-1]=='-' else x for x in Origin]; 
Origin  = ['99' if x=='2A' else x for x in Origin]; 
Origin  = ['100' if x=='2B' else x for x in Origin]; 
Origin = list(map(int, Origin));
Origin= np.asarray(Origin);

# 
Finess = [w.replace('2A', '99') for w in Finess]
Finess = [w.replace('2B', '100') for w in Finess]
Finess = list(map(int, Finess));


FileLoc = FilePath + 'DataTrain'
np.savez_compressed(FileLoc, Finess=Finess, Origin=Origin, Speciality=Speciality, Age=Age,SejourALD=SejourALD,Sejourtot=Sejourtot,Year=Year,Target=Target)



