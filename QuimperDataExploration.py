# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 2016

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
FiOrigin = HospitalCounty(Finess)
Proxy = np.zeros(len(Origin)); Proxy[(FiOrigin == Origin)] = 1;

########################################################################
# Hospital location influence : CHIC DE CORNOUAILLE QUIMPER

QUIMPERTarget = Target[Finess == 290020700]
QUIMPERAge = Age[Finess == 290020700]
QUIMPERSejourALD = SejourALD[Finess == 290020700]
QUIMPERSejourtot = Sejourtot[Finess == 290020700]
QUIMPEROrigin = Origin[Finess == 290020700]
QUIMPERYear = Year[Finess == 290020700]
QUIMPERSpeciality = Speciality[Finess == 290020700]
QUIMPERratioALD = ratioALD[Finess == 290020700]
QUIMPERProxy = Proxy[Finess == 290020700]

np.corrcoef(QUIMPERTarget,QUIMPERSejourALD)[1,0] # 0.068741224705110723
np.corrcoef(QUIMPERTarget,QUIMPERSejourtot)[1,0] # 0.041430739915669978
np.corrcoef(QUIMPERTarget,QUIMPEROrigin)[1,0]    # 0.048317081770126308
np.corrcoef(QUIMPERTarget,QUIMPERYear)[1,0]      # 0.27701965547902696
np.corrcoef(QUIMPERTarget,QUIMPERAge)[1,0]       # 0.30524871298682754
np.corrcoef(QUIMPERTarget,QUIMPERSpeciality)[1,0]# 0.2193390299578992
np.corrcoef(QUIMPERTarget,QUIMPERratioALD)[1,0]  # 0.77766247725114412

########################################################################
# Influence of the distance between the Hospital and patient county
Dist = np.ones(len(np.unique(QUIMPEROrigin)))*2; region =[35,56,22];
for x in np.arange(0,len(Dist)):
     if  np.unique(QUIMPEROrigin)[x]  == 29: 
            Dist[x] = 0
     if  np.unique(QUIMPEROrigin)[x] in region: 
            Dist[x] = 1
     if  np.unique(QUIMPEROrigin)[x]  == 0:     
            Dist[x] = 3

OriginDist = np.zeros(len(QUIMPEROrigin));
for x in np.arange(0,len(QUIMPEROrigin)):
     OriginDist[x] = Dist[(np.unique(QUIMPEROrigin) == QUIMPEROrigin[x])]

np.corrcoef(QUIMPERTarget,OriginDist)[1,0] 

plt.scatter(QUIMPERratioALD[:,0],QUIMPERTarget[:,0],s=QUIMPERTarget[:,1]*10000); 	
plt.errorbar(QUIMPERratioALD[:,0],QUIMPERTarget[:,0],yerr=QUIMPERTarget[:,2])
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Variable Cible',size=16);
plt.grid(True); plt.show();

########################################################################
# Multi-variables analysis

colors = cm.rainbow(np.linspace(0, 1, 7))
for x in np.arange(0,5):
    X = QUIMPERratioALD[np.where((QUIMPERAge==1) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==1) )]
    Y = QUIMPERTarget[np.where( (QUIMPERAge==1) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==1) )]
    plt.scatter(X,Y, marker='8', s=60, color=colors[x]); 
    X = QUIMPERratioALD[np.where((QUIMPERAge==1) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==0) )]
    Y = QUIMPERTarget[np.where( (QUIMPERAge==1) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==0) )]
    plt.scatter(X,Y, marker='8', s=20, color=colors[x]); 
    X = QUIMPERratioALD[np.where((QUIMPERAge==0) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==1) )]
    Y = QUIMPERTarget[np.where( (QUIMPERAge==0) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==1) )]
    plt.scatter(X,Y, marker='v', s=60, color=colors[x]); 
    X = QUIMPERratioALD[np.where((QUIMPERAge==0) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==0) )]
    Y = QUIMPERTarget[np.where( (QUIMPERAge==0) & (QUIMPERYear==(x+2008))  & (QUIMPERProxy==0) )]
    plt.scatter(X,Y, marker='v', s=20, color=colors[x]); 
	
plt.title('Centre hospitalier de Cornouaille Quimper',size=20); 
plt.xlabel('Ratio séjour ALD / séjour total',size=16); plt.ylabel('Variable Cible',size=16);
plt.grid(True); plt.show();






