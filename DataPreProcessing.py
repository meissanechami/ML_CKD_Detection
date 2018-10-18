from DecisionTree import classifyDT, plotHP, reportDT, plotDT, selection
from GaussianNB import func,reportNB
from SVMcl import reportSVM
from RandomForest import reportRF
import pandas as pd
from itertools import combinations
import scipy.stats as stats
import pylab as plt
from DataFrameImputer import DataFrameImputer
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.cross_validation import cross_val_score
import numpy as np
import random
from CValidated import kfoldCV
from kfolds_DT import myownStratifiedKfolds,balanced_subsample,small_regression
from Multiplerun import MultipleRuns
from Ratio import Ratio
from Surrogatedistribution import distribution
from optimisationMS import optimisationML
import itertools
import numpy as np
import pylab as p
import scipy as s
import scipy.stats as stats
from costAnalysis import cost_analysis
from DataFrameImputer import DataFrameImputer


class dataSetup():
    
    def __init__(self):
        pass    
    
    def dataLoading(self,file):
        with open(file) as f:
            features = []
            labels = []
            for line in f:
                line = line.replace("@data","")
                line = line.replace("yes","0")
                line = line.replace("no,","1,")
                line = line.replace("poor","0")
                line = line.replace("good","1")
                line = line.replace(",normal",",1")
                line = line.replace(",abnormal",",0")
                line = line.replace(",notpresent",",1")
                line = line.replace(",present",",0")
                line = line.replace("?", "NaN")
                line = line.replace(",notckd",",1")
                line = line.replace(",ckd",",0")
                line = line.replace("\t","")
                line = line[:-2]
                row = line.split(",")
                labels.append(row[-1])
                features.append(row[:-1])       
            features = features[8:]
            labels = labels[8:]
            labels[399] = "1"
            features = [list(func(i)) for i in features]
            labels = [list(func(i)) for i in labels]
            labels = np.asarray(labels)
            labels = labels.reshape((400,))
            features = [pd.to_numeric(i,errors="coerce") for i in features]
            features_before = pd.DataFrame(features)
            features_after = DataFrameImputer().fit_transform(features_before)
        return features_after,labels
        
#************************* Main **********************       
if __name__ == '__main__' :    
#************************* Pre-processing the data from text file **********************************    
    setup = dataSetup()
    features , labels = setup.dataLoading("/Users/meissanechami/Desktop/MachineLearning/DataSets.RTF")
#**************** Stratified K-fold on optimised models ********************
    kfoldCV (features, labels)    
#*********************************** Bayesian optimisation ***************************************
    def svccv(C):
        return cross_val_score(SVC(C=C, kernel='linear', random_state=2),
                                features.values, labels, cv=5).mean()
                                
    def rfccv(n_estimators, min_samples_split, max_features):
        return cross_val_score(RFC(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               random_state=2),
                           features.values, labels, cv=5).mean()
                           
    def dtccv(min_samples_split, max_features):
        return cross_val_score(DT(min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               random_state=2),
                           features.values, labels, cv=5).mean()
    
                    
    rfcBO = BayesianOptimization(rfccv, {'n_estimators': (10, 250),
                                         'min_samples_split': (2, 25),
                                         'max_features': (1,24)})
                                         
    dtcBO = BayesianOptimization(dtccv, {'min_samples_split': (2, 25),
                                         'max_features': (1,24)})
    
    svcBO = BayesianOptimization(svccv, {'C': (0.001, 0.1)})
    
    svcBO.explore({'C': [0.001, 0.01, 0.1]})
    
    rfcBO.maximize()
    svcBO.maximize()
    dtcBO.maximize()

    print('-'*53)
    print('Final Results')
    print('RF: %f' % rfcBO.res['max']['max_val'])
    print('SVM: %f' % svcBO.res['max']['max_val'])
    print('DT values: %f' % dtcBO.res['max']['max_val'])
    print('NB: %f' % cross_val_score(NB(),features.values, labels, cv=5).mean())   
#*********************************** Multiple Runs ***************************************
    MSs = MultipleRuns()
    MSs.MS_nonoptimised(features, labels)
    
    MtS = optimisationML()
    MtS.MS(features,labels)
#******************* parameter importances on validation set *******************************

    myownStratifiedKfolds().kfold_DT(features,labels,5)  
#********************************** Suroggate Data Test *************************************
    dis = distribution()
    dis.trace_gaussian(features,labels,50)
#**************************************** Ratio Test ***************************************
    ratio = Ratio()
    ratio.plot_ratio(features,labels)   
#**************************************** END ***************************************

 
 
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
    
    
    
    
    
