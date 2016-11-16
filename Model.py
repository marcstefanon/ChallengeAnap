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
# Linear Model
###############
# Ridge regression
###### Training Data ######
Y = Target;
X = np.vstack((Age,Origin,Finess,Year,Speciality,SejourALD,ratioALD)).transpose()
clf = linear_model.Ridge (alpha = .5, normalize=False)
clf.fit(X, Y); TrainRes = clf.predict(X); # TrainRes[SejourALD == 0] = 0;
r2_score_ridge = r2_score(Y, TrainRes)  
rmse(Y,TrainRes )
nrmse(Y,TrainRes)
np.corrcoef(Y, TrainRes)[1,0]
# >>> rmse(Y, TrainRes) 0.094700164205765533
# >>> nrmse(Y, TrainRes) 0.077942892578020825
# >>> np.corrcoef(Y, TrainRes)[1,0] 0.76774531565599913

###### Test Data ######
X = np.vstack((TestAge,TestOrigin,TestFiness,TestYear,TestSpeciality,TestSejourALD,TestratioALD)).transpose()
Res = np.sum(clf.coef_ * X,1) + clf.intercept_

WriteRes(Res)

###### Graphics for Training Data ######
NRes = aggregateRes(100,TrainRes,ratioALD)

plt.scatter(NratioALD[:,0],NTarget[:,0],s=NTarget[:,1]*10000,c='b',label='Target',alpha=0.5); 
plt.scatter(NratioALD[:,0],NRes[:,0],s=NRes[:,1]*10000,c='r',label='Model'); 	
plt.errorbar(NratioALD[:,0],NTarget[:,0],yerr=NTarget[:,2])
plt.errorbar(NratioALD[:,0],NRes[:,0],yerr=NRes[:,2], ecolor='r')
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Variable Cible',size=16);
plt.legend(loc=4, scatterpoints=1,markerscale=0.4)
plt.title("Linear Model \n r$^2$ determination = " + str(round(r2_score(TrainRes, Target),2)) + "    r$^2$ correlation = " + str(round(np.corrcoef(TrainRes,Target)[0,1],2)) + "    RMSE = "+ str(round(rmse(TrainRes, Target),2)),size=20)
plt.grid(True); plt.show();


##############################################################################################################
# Piecewise linear Model

###### Training Data ######
# Sorting depends on ratioALD values above or below 0.5
P1Target = Target[ratioALD < 0.51]; P1Age = Age[ratioALD < 0.51];
P1Origin = Origin[ratioALD < 0.51]; P1Finess = Finess[ratioALD < 0.51];
P1Year = Year[ratioALD < 0.51]; P1Speciality = Speciality[ratioALD < 0.51];
P1SejourALD = SejourALD[ratioALD < 0.51]; P1ratioALD = ratioALD[ratioALD < 0.51];
P1Proxy = Proxy[ratioALD < 0.51];

P2Target = Target[ratioALD >= 0.51]; P2Age = Age[ratioALD >= 0.51];
P2Origin = Origin[ratioALD >= 0.51]; P2Finess = Finess[ratioALD >= 0.51];
P2Year = Year[ratioALD >= 0.51]; P2Speciality = Speciality[ratioALD >= 0.51];
P2SejourALD = SejourALD[ratioALD >= 0.51]; P2ratioALD = ratioALD[ratioALD >= 0.51];
P2Proxy = Proxy[ratioALD >= 0.51];

# Fitting model for the first part
Y = P1Target; X = np.vstack((P1Age,P1Origin,P1Finess,P1Year,P1Speciality,P1SejourALD,P1ratioALD,P1Proxy)).transpose()
P1clf = linear_model.Ridge (alpha = .5, normalize=False)
P1clf.fit(X, Y); TrainRes = P1clf.predict(X); 
r2_score_ridge = r2_score(Y, TrainRes)  
rmse(Y,TrainRes)
nrmse(Y,TrainRes)
np.corrcoef(Y, TrainRes)[1,0] 
# >>> rmse(Y, TrainRes) 0.072148969354404319  | 0.077188221359039508
# >>> nrmse(Y, TrainRes) 0.049350666442301259 | 0.04695509329129384
# >>> np.corrcoef(Y, TrainRes)[1,0] 0.79106060305811354 | 0.74094025878701086


# Fitting model for the second part
Y = P2Target; X = np.vstack((P2Age,P2Origin,P2Finess,P2Year,P2Speciality,P2SejourALD,P2ratioALD,P2Proxy)).transpose()
P2clf = linear_model.Ridge (alpha = .5, normalize=False)
P2clf.fit(X, Y); TrainRes = P2clf.predict(X); 
r2_score_ridge = r2_score(Y, TrainRes)  
rmse(Y,TrainRes)
nrmse(Y,TrainRes)
np.corrcoef(Y, TrainRes)[1,0]
# >>> rmse(Y, TrainRes) 0.095005044226791208 | 0.092619451766436267
# >>> nrmse(Y, TrainRes) 0.16196088932633562 | 0.25758713452880827
# >>> np.corrcoef(Y, TrainRes)[1,0] 0.4620996459182527 | 0.35734843893250801


# Model Assembly
X = np.vstack((P1Age,P1Origin,P1Finess,P1Year,P1Speciality,P1SejourALD,P1ratioALD,P1Proxy)).transpose()
P1Res = np.sum(P1clf.coef_ * X,1) + P1clf.intercept_
X = np.vstack((P2Age,P2Origin,P2Finess,P2Year,P2Speciality,P2SejourALD,P2ratioALD,P2Proxy)).transpose()
P2Res = np.sum(P2clf.coef_ * X,1) + P2clf.intercept_

TrainRes = np.zeros(len(ratioALD)); TrainRes[ratioALD < 0.51] = P1Res;  TrainRes[ratioALD >= 0.51] = P2Res; Y = Target;
rmse(Y,TrainRes) 0.08412003530117966
nrmse(Y,TrainRes) 0.059628781967145023
np.corrcoef(Y, TrainRes)[1,0] 0.82222095501857961


###### Graphics for Training Data ######
NRes = aggregateRes(100,TrainRes,ratioALD)

plt.scatter(NratioALD[:,0],NTarget[:,0],s=NTarget[:,1]*10000,c='b',label='Target',alpha=0.5); 
plt.scatter(NratioALD[:,0],NRes[:,0],s=NRes[:,1]*10000,c='r',label='Model'); 	
plt.errorbar(NratioALD[:,0],NTarget[:,0],yerr=NTarget[:,2])
plt.errorbar(NratioALD[:,0],NRes[:,0],yerr=NRes[:,2], ecolor='r')
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Variable Cible',size=16);
plt.legend(loc=4, scatterpoints=1,markerscale=0.4)
plt.title("Linear Model \n r$^2$ determination = " + str(round(r2_score(TrainRes, Target),2)) + "    r$^2$ correlation = " + str(round(np.corrcoef(TrainRes,Target)[0,1],2)) + "    RMSE = "+ str(round(rmse(TrainRes, Target),2)),size=20)
plt.grid(True); plt.show();

###### Test Data ######
# Sorting depends on ratioALD values above or below 0.5
P1TestAge = TestAge[TestratioALD < 0.51]; P1TestProxy = TestProxy[TestratioALD < 0.51];
P1TestOrigin = TestOrigin[TestratioALD < 0.51]; P1TestFiness = TestFiness[TestratioALD < 0.51];
P1TestYear = TestYear[TestratioALD < 0.51]; P1TestSpeciality = TestSpeciality[TestratioALD < 0.51];
P1TestSejourALD = TestSejourALD[TestratioALD < 0.51]; P1TestratioALD = TestratioALD[TestratioALD < 0.51];

P2TestAge = TestAge[TestratioALD >= 0.51]; P2TestProxy = TestProxy[TestratioALD >= 0.51];
P2TestOrigin = TestOrigin[TestratioALD >= 0.51]; P2TestFiness = TestFiness[TestratioALD >= 0.51];
P2TestYear = TestYear[TestratioALD >= 0.51]; P2TestSpeciality = TestSpeciality[TestratioALD >= 0.51];
P2TestSejourALD = TestSejourALD[TestratioALD >= 0.51]; P2TestratioALD = TestratioALD[TestratioALD >= 0.51];

###### Model Assembly ######
X = np.vstack((P1TestAge,P1TestOrigin,P1TestFiness,P1TestYear,P1TestSpeciality,P1TestSejourALD,P1TestratioALD,P1TestProxy)).transpose()
P1Res = np.sum(P1clf.coef_ * X,1) + P1clf.intercept_
X = np.vstack((P2TestAge,P2TestOrigin,P2TestFiness,P2TestYear,P2TestSpeciality,P2TestSejourALD,P2TestratioALD,P2TestProxy)).transpose()
P2Res = np.sum(P2clf.coef_ * X,1) + P2clf.intercept_

Res = np.zeros(len(TestratioALD)); Res[TestratioALD < 0.51] = P1Res;  Res[TestratioALD >= 0.51] = P2Res;

###### Writing results to  csv files (challenge format) ######
WriteRes(Res)


##############################################################################################################
# Truncated Sigmoide 
def AnapFct(X,K,L,a0):
    A = K / (1 + np.exp(-L*X)) 
    return a0 + A 

popt, pcov = curve_fit(AnapFct, ratioALD, Target)
TrainRes = AnapFct(ratioALD,popt[0],popt[1],popt[2])

###### Graphics for Training Data ######
NRes = aggregateRes(100,TrainRes,ratioALD)

plt.scatter(NratioALD[:,0],NTarget[:,0],s=NTarget[:,1]*10000,c='b',label='Target',alpha=0.5); 
plt.scatter(NratioALD[:,0],NRes[:,0],s=NRes[:,1]*10000,c='r',label='Model'); 	
plt.errorbar(NratioALD[:,0],NTarget[:,0],yerr=NTarget[:,2])
plt.errorbar(NratioALD[:,0],NRes[:,0],yerr=NRes[:,2], ecolor='r')
plt.xlabel('Ratio sejour ALD / sejour total',size=16); plt.ylabel('Variable Cible',size=16);
plt.legend(loc=4, scatterpoints=1,markerscale=0.4)
plt.title("Linear Model \n r$^2$ determination = " + str(round(r2_score(TrainRes, Target),2)) + "    r$^2$ correlation = " + str(round(np.corrcoef(TrainRes,Target)[0,1],2)) + "    RMSE = "+ str(round(rmse(TrainRes, Target),2)),size=20)
plt.grid(True); plt.show();

