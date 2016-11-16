# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 2016

@author: Marc Stefanon
"""
# Sources
# http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
########################################################################
from ChallengeAnap import *

########################################################################
# Loading training data
(Target,Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality) = loadTrain()

# Loading training data
(TestAge,TestSejourALD,TestSejourtot,TestFiness,TestOrigin,TestYear,TestSpeciality) = loadTest()
TestratioALD = TestSejourALD / TestSejourtot;


########################################################################
# Removing case where Target = 0 since it is conditionned by SejourALD = 0

TotaLength = len(SejourALD)
Target = Target[SejourALD >0]; 
Age = Age[SejourALD >0];
Origin = Origin[SejourALD >0];
Sejourtot = Sejourtot[SejourALD >0]; 
Finess = Finess[SejourALD >0];
Year = Year[SejourALD >0];
Speciality = Speciality[SejourALD >0];
SejourALD = SejourALD[SejourALD >0];
ratioALD = SejourALD / Sejourtot;
100*len(SejourALD) / TotaLength

# Computing Hospital County number
FiOrigin = HospitalCounty(Finess)
TestFiOrigin = HospitalCounty(TestFiness)

# Computing Proximity to patient's home 
Proxy = np.zeros(len(Origin)); Proxy[(FiOrigin == Origin)] = 1;
TestProxy = np.zeros(len(TestOrigin)); TestProxy[(TestFiOrigin == TestOrigin)] = 1;

# Data aggregation by bins
(NTarget,NAge,NSejourALD,NSejourtot,NFiness,NOrigin,NYear,NSpeciality,NratioALD) = aggregate(100,Target,Age,SejourALD,Sejourtot,Finess,Origin,Year,Speciality,ratioALD)

########################################################################
# Neural network

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

Y = Target; Y = list(Target)
X = np.vstack((Age,Origin,Finess,Year,Speciality,SejourALD,ratioALD,Proxy)).transpose()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

mlp = MLPRegressor(activation='relu',hidden_layer_sizes=(30,30,30))
mlp.fit(X, Y) 
TrainRes = mlp.predict(X)

print(rmse(Target,TrainRes))
print(nrmse(Target,TrainRes))
print(np.corrcoef(Target, TrainRes)[1,0])
print(r2_score(TrainRes, Target))


###### Graphics for Training Data ######
NRes = aggregateRes(100,TrainRes,ratioALD)

plt.scatter(NratioALD[:,0],NTarget[:,0],s=NTarget[:,1]*10000,c='b',label='Target',alpha=0.5); 
plt.scatter(NratioALD[:,0],NRes[:,0],s=NRes[:,1]*10000,c='r',label='Model'); 	
plt.errorbar(NratioALD[:,0],NTarget[:,0],yerr=NTarget[:,2])
plt.errorbar(NratioALD[:,0],NRes[:,0],yerr=NRes[:,2], ecolor='r')
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Variable Cible',size=16);
plt.legend(loc=4, scatterpoints=1,markerscale=0.4)
plt.title("Neural Network \n r$^2$ determination = " + str(round(r2_score(TrainRes, Target),2)) + "    r$^2$ correlation = " + str(round(np.corrcoef(TrainRes,Target)[0,1],2)) + "    RMSE = "+ str(round(rmse(TrainRes, Target),2)),size=20)
plt.grid(True); plt.show();



