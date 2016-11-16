# -*- coding: utf-8 -*-
"""
Created on Wed Nov 4 2016

@author: Marc Stefanon
"""
# Sources
########################################################################
from ChallengeAnap import *

########################################################################
# Loading training data
(Target,Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality) = loadTrain()

########################################################################
# Removing case where Target = 0 since it is conditionned by SejourALD = 0

TotaLength = len(SejourALD)
Target = Target[SejourALD >0]; Age = Age[SejourALD >0];
Origin = Origin[SejourALD >0]; Sejourtot = Sejourtot[SejourALD >0]; 
Finess = Finess[SejourALD >0]; Year = Year[SejourALD >0];
Speciality = Speciality[SejourALD >0]; SejourALD = SejourALD[SejourALD >0];
ratioALD = SejourALD / Sejourtot;
100*len(SejourALD) / TotaLength

# Computing Hospital County number
FiOrigin = HospitalCounty(Finess):

########################################################################
# Some general correlation

np.corrcoef(Target,SejourALD)[1,0] # 0.058752583076026377
np.corrcoef(Target,Sejourtot)[1,0] # -0.013989790113634203
np.corrcoef(Target,Origin)[1,0]    # 0.0037079972469922016
np.corrcoef(Target,Finess)[1,0]    # 0.010962628591294361
np.corrcoef(Target,Year)[1,0]      # 0.19215906745775937
np.corrcoef(Target,Age)[1,0]       # 0.21815879031267732
np.corrcoef(Target,Speciality)[1,0]# 0.11179765569311094
np.corrcoef(Target,ratioALD)[1,0]  # 0.73950908183490882


Target = Target[ratioALD !=1]; Age = Age[ratioALD !=1]; 
Origin = Origin[ratioALD !=1]; Sejourtot = Sejourtot[ratioALD !=1]; 
Finess = Finess[ratioALD !=1]; Year = Year[ratioALD !=1]; 
Speciality = Speciality[ratioALD !=1]; SejourALD = SejourALD[ratioALD !=1]; 
ratioALD = SejourALD / Sejourtot;

np.corrcoef(Target,SejourALD)[1,0] # 0.11439617368574385
np.corrcoef(Target,Sejourtot)[1,0] # 0.057048769609457865
np.corrcoef(Target,Origin)[1,0]    # -0.027614286639183128
np.corrcoef(Target,Finess)[1,0]    # -0.00043870444794695681
np.corrcoef(Target,Year)[1,0]      # 0.16666130940907048
np.corrcoef(Target,Age)[1,0]       # 0.28392775374196227
np.corrcoef(Target,Speciality)[1,0]# 0.13216093334535348
np.corrcoef(Target,ratioALD)[1,0]  # 0.76623433434801691


(Target,Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality) = loadTrain()
ratioALD = SejourALD / Sejourtot;
Target = Target[ratioALD ==1]; Age = Age[ratioALD ==1]; 
Origin = Origin[ratioALD ==1]; Sejourtot = Sejourtot[ratioALD ==1]; 
Finess = Finess[ratioALD ==1]; Year = Year[ratioALD ==1]; 
Speciality = Speciality[ratioALD ==1]; SejourALD = SejourALD[ratioALD ==1]; 
ratioALD = SejourALD / Sejourtot;


np.corrcoef(Target,SejourALD)[1,0] # -0.020759720391103312
np.corrcoef(Target,Sejourtot)[1,0] # -0.020759720391103312
np.corrcoef(Target,Origin)[1,0]    # -0.0056506479186578373
np.corrcoef(Target,Finess)[1,0]    # 0.053213966176474531
np.corrcoef(Target,Year)[1,0]      # 0.35052425740692217
np.corrcoef(Target,Age)[1,0]       # 0.015518402752959378
np.corrcoef(Target,Speciality)[1,0]# -0.066173730904294417
np.corrcoef(Target,ratioALD)[1,0]  # nan


########################################################################
# Study Age influence

np.mean(Target[np.where( Age==0 )])
np.mean(Target[np.where( Age==1 )])

########################################################################
# Study ratio influence
plt.hist(ratioALD,100); plt.title("Ratio of hospitalisation related to ALD \n over the  total number of hospitalisation"); plt.show();
plt.hist(np.log(test),100); plt.show()


########################################################################
# Study hospital services influence

SpeName = []
SpeName.append('Digestif'); SpeName.append('Orthopédie'); SpeName.append('Traumatismes'); SpeName.append('Rhumatologie');
SpeName.append('Système nerveux'); SpeName.append('Cathétérismes vasculaires'); SpeName.append('Cardio-vasculaire'); SpeName.append('Vasculaire périphérique');
SpeName.append('Pneumologie'); SpeName.append('ORL Stomatologie'); SpeName.append('Gynécologie Sein'); SpeName.append('Obstétrique');
SpeName.append('Nouveau-nés et période périnatale'); SpeName.append('Uro-néphrologie et génital'); SpeName.append('Hématologie'); 
SpeName.append('Chimiothérapie radiothérapie'); SpeName.append('Maladies infectieuses'); SpeName.append('Endocrinologie'); SpeName.append('Tissu cutané et sous-cutané'); 
SpeName.append('Brûlures'); SpeName.append('Psychiatrie'); SpeName.append('Toxicologie, Intoxications	, Alcool'); SpeName.append('Douleurs chroniques, Soins palliatifs');
SpeName.append('Transplantion'); SpeName.append('Orthopédie'); SpeName.append('Séances'); SpeName.append('Rhumatologie');

SpeRanking = []
for x in np.arange(1,np.max((Speciality))):
    SpeRanking.append(np.mean(Target[np.where(Speciality==x)]))

np.array(SpeRanking)[np.argsort(SpeRanking)]
np.array(SpeName)[np.argsort(SpeRanking)]

np.mean(Target[np.where( (Speciality==np.argsort(SpeRanking)[-2]+1) )])

# Case where Hospitals Stay are fully caused by ALD
TempTarget = Target[ratioALD == 1]; TempSpeciality = Speciality[ratioALD == 1]; 
SpeRanking = []
for x in np.arange(1,np.max((Speciality))):
    SpeRanking.append(np.mean(TempTarget[np.where(TempSpeciality==x)]))

np.array(SpeRanking)[np.argsort(SpeRanking)]
np.array(SpeName)[np.argsort(SpeRanking)]	

########################################################################
# Study year influence 

YearTarget = []; YearatioALD = []; YearSejourALD = [];
YearAge = []; YearSpe = []; 
for x in np.arange(2008,2014):
    YearTarget.append(Target[Year==x].mean())
    YearatioALD.append(ratioALD[Year==x].mean())
    YearSejourALD.append(SejourALD[Year==x].mean())
    YearAge.append(Age[Year==x].mean())
    YearSpe.append(Speciality[Year==x].mean())

np.corrcoef(YearatioALD,YearTarget)[1,0]   # 0.94810898478486694
np.corrcoef(YearSejourALD,YearTarget)[1,0] # 0.90615337064695622
np.corrcoef(YearAge,YearTarget)[1,0]       # 0.59261485319644991
np.corrcoef(YearSpe,YearTarget)[1,0]       # 0.38974937896130857

########################################################################
# Multi-variables analysis
(NTarget,NAge,NSejourALD,NSejourtot,NFiness,NOrigin,NYear,NSpeciality,NratioALD) = aggregate(100,Target,Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality,ratioALD)

#
plt.scatter(NratioALD[:,0],NTarget[:,0],s=NTarget[:,1]*10000); 	
plt.errorbar(NratioALD[:,0],NTarget[:,0],yerr=NTarget[:,2])
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Variable Cible',size=16);
plt.grid(True); plt.show();
plt.scatter(NratioALD[:,0],NSejourALD[:,0],s=NSejourALD[:,1]*10000); 
plt.errorbar(NratioALD[:,0],NSejourALD[:,0],yerr=NSejourALD[:,2])
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Nombre de sejours ALD',size=16);
plt.grid(True); plt.show();
plt.scatter(NratioALD[:,0],NSpeciality[:,0],s=NSpeciality[:,1]*10000); 
plt.errorbar(NratioALD[:,0],NSpeciality[:,0],yerr=NSpeciality[:,2])
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Specialite',size=16);
plt.grid(True); plt.show();
plt.scatter(NratioALD[:,0],NAge[:,0],s=NAge[:,1]*10000); 
plt.errorbar(NratioALD[:,0],NAge[:,0],yerr=NAge[:,2])
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Age',size=16);
plt.grid(True); plt.show();
plt.scatter(NratioALD[:,0],NYear[:,0],s=NYear[:,1]*10000);  plt.ylim([2010,2011])
plt.errorbar(NratioALD[:,0],NYear[:,0],yerr=NYear[:,2])
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Annee',size=16);
plt.grid(True); plt.show();

########################################################################
# Behavior when ratioALD == 1

UnTarget = Target[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnAge = Age[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnOrigin = Origin[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnSejourtot = Sejourtot[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnFiness = Finess[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnYear = Year[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnSpeciality =  Speciality[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnSejourALD =  SejourALD[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]
UnratioALD =  ratioALD[np.logical_and(ratioALD > 0.99,ratioALD <= 1)]


plt.scatter(UnratioALD,UnTarget); plt.show() 	
plt.scatter(UnAge,UnTarget); plt.show() 	
plt.scatter(UnOrigin,UnTarget); plt.show() 
plt.scatter(UnSejourALD,UnTarget); plt.show() 		
plt.scatter(UnSejourtot,UnTarget); plt.show() 	
plt.scatter(UnFiness,UnTarget); plt.show() 	
plt.scatter(UnYear,UnTarget); plt.show() 	
plt.scatter(UnSpeciality,UnTarget); plt.show() 

float(UnSejourtot[UnSejourtot>10000].sum()) / UnSejourtot.sum()

DeratioALD = UnratioALD[UnSejourtot>10000]
DeAge = UnAge[UnSejourtot>10000]
DeOrigin = UnOrigin[UnSejourtot>10000]
DeSejourALD = UnSejourALD[UnSejourtot>10000]
DeSejourtot = UnSejourtot[UnSejourtot>10000]
DeFiness = UnFiness[UnSejourtot>10000]
DeYear = UnYear[UnSejourtot>10000]
DeSpeciality = UnSpeciality[UnSejourtot>10000]
DeTarget = UnTarget[UnSejourtot>10000]

plt.scatter(DeratioALD,DeTarget); plt.show() 	
plt.scatter(DeAge,DeTarget); plt.show() 	
plt.scatter(DeOrigin,DeTarget); plt.show() 
plt.scatter(DeSejourALD,DeTarget); plt.show() 		
plt.scatter(DeSejourtot,DeTarget); plt.show() 	
plt.scatter(DeFiness,DeTarget); plt.show() 	
plt.scatter(DeYear,DeTarget); plt.show() 	
plt.scatter(DeSpeciality,DeTarget); plt.show() 



