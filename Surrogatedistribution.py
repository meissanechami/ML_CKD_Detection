from DecisionTree import classifyDT, plotHP, reportDT
from GaussianNB import func
import pandas as pd
from itertools import combinations
import scipy.stats as stats
import pylab as pl
from DataFrameImputer import DataFrameImputer
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.cross_validation import cross_val_score
import numpy as np
import random
from Ratio import Ratio

class distribution():
    
    def __init__(self):
        self.lis = [14, 2, 17, 3, 12, 16, 13, 15]
    
    def trace_distribution(self, features_after, labels, u):
        ScoreDT = []
        DToriginal = []
        sampledfeatures = features_after.sample(u)
        index = sampledfeatures.index.tolist()
    
        combi = list(combinations(self.lis,5))
        pl.figure(facecolor='white')
        for x in combi:
            features = sampledfeatures.loc[:,x]
            print(features)
            features_origin = sampledfeatures.loc[:,x]
            print(features_origin)
            features = features.reset_index(drop=True)
            features.loc[:,x] = Ratio().shuffle(features.loc[:,x])
            label = []
            for n in index:
                label.append(labels[n]) 
            
            clfTestDT = cross_val_score(DT(min_samples_split=5,
                               random_state=2),features.values, label, cv=5).mean()
            ScoreDT.append(clfTestDT)
        
            clfTestDTorigin = cross_val_score(DT(min_samples_split=5,
                               random_state=2),features_origin.values, label, cv=5).mean()
            DToriginal.append(clfTestDTorigin)  
            
        h = sorted(ScoreDT)
        fit = stats.norm.pdf(h, np.mean(h), np.std(h))  
       # pl.plot(h,fit)#,label='Surrogates: mean=%0.2f'% np.mean(h))
        pl.hist(h,normed=True,label='Surrogates: mean=%0.2f'% np.mean(h))  
    
        v = sorted(DToriginal)
        fit1 = stats.norm.pdf(v, np.mean(v), np.std(v))  
        #pl.plot(v,fit1)#'-o',label='Real data: mean=%0.2f'% np.mean(v))
        pl.hist(v,normed=True,label='Real data: mean=%0.2f'% np.mean(v)) 
        pl.legend(bbox_to_anchor=(0., -0.12, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.) 
        pl.title('Surrogate data testing for %s random uniform samples and 5 features'% u)
        pl.show()
        print (np.mean(h), np.std(h))
        print (np.mean(v), np.std(v))


            
            
        
                