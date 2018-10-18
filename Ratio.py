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


class Ratio():
    
    def __init__(self):
        self.sub = 331
        self.lis = [14, 2, 17, 3, 12, 16, 13, 15]
        self.setSize = [20,40,50,60,70,80,90,100,400]
        self.featuresArray = [2,3,4,5,6,7,8]
        
    def shuffle(self,df, n=1, axis=0):
        df=df.copy()
        for k in range(n):
            df.apply(np.random.shuffle,axis=axis,reduce=None)
        return df
        
    def plot_ratio(self, features_after, labels):
        Allspaces = []
        pl.figure(facecolor='white',frameon='black')
        for set in self.setSize:
            ax1 = pl.subplot(self.sub)
            ax2 = ax1.twinx()
            ax1.grid(False)
            ax2.grid(False)
            pl.title('Surrogate test on %s random samples'%set)
            ax1.set_xlim(xmin=0,xmax=(set/2)+(set/20))
            ax2.set_ylim(ymin=-0.5,ymax=0.7)
            ax1.set_ylim(ymin=-0,ymax=1)
            ax2.plot([0,(set/2)+(set/20)],[0,0],'w--')
            ax2.set_ylabel('∆R')
            ax1.set_ylabel('R_low')
            ax1.set_xlabel('Ratio φ')
              
            sampledfeatures = features_after.sample(set)
            index = sampledfeatures.index.tolist()
            rto=[]
            gap = []
            rlow=[]
            maspace = []
            for feat in self.featuresArray:
                ScoreDT = []
                DToriginal = []
                combi = list(combinations(self.lis,feat))
                space = []
                for x in combi:
                    features = sampledfeatures.loc[:,x]
                    features_origin = sampledfeatures.loc[:,x]
                    features = features.reset_index(drop=True)
                    features.loc[:,x] = self.shuffle(features.loc[:,x])
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
    
                v = sorted(DToriginal)
                fit1 = stats.norm.pdf(v, np.mean(v), np.std(v))
               
            
                difference = min(DToriginal) - max(ScoreDT)
                print(difference)
                space.append('set size %s feature %s'%(set,feat))
                space.append(difference)
                gap.append(difference)
                ratio = set/feat
                rto.append(ratio)
                space.append('ratio is %s'%ratio)
                maspace.append(space) 
                  
                r=set/55
                if difference >= 0:
                    ax1.plot(ratio,min(DToriginal),'o',fillstyle='full',markeredgewidth=0.01)
                    rlow.append(min(DToriginal))
                if difference < 0 :
                    ax1.plot(ratio,max(ScoreDT), 'o',fillstyle='full',markeredgewidth=0.01)
                    rlow.append(max(ScoreDT))
                
                ax2.plot(ratio,difference,'o',fillstyle='full',markeredgewidth=0.01,label=' %s features'%feat)

            ax1.plot(rto,rlow,'g-',label='R_low')
            ax2.plot(rto,gap,'r-',label='∆R')
            self.sub = self.sub + 1   
            Allspaces.append(maspace)
        ax1.legend(loc='best',bbox_to_anchor=(1.8, 1.2),fancybox=True, shadow=True)
        ax2.legend(loc='best',bbox_to_anchor=(1.8, 0.2),fancybox=True, shadow=True)
        pl.show()
        print(Allspaces)
