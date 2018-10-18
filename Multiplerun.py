from GaussianNB import func
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.cross_validation import cross_val_score

class MultipleRuns():

    def __init__(self):
        self.Tests = [30,50,70,90,
                110,130,150,170,
                190,210,230,250,
                270,290,310,330,
                350,370,400]

    def MS_nonoptimised(self, features_after, labels):        
        for x in self.Tests:
            ScoreDT = []
            ScoreSVM = []
            ScoreRF = []
            ScoreNB = []
            for i in range(100):
                featuresTraining = features_after.sample(int(x))
                indexTraining = featuresTraining.index.tolist()
                labelsTraining = []
                for n in indexTraining:
                    labelsTraining.append(labels[n])      
         
                clfTestDT = cross_val_score(DT(),
                               featuresTraining, labelsTraining, cv=5).mean()
                ScoreDT.append(clfTestDT)
            
                clfTestSVM = cross_val_score(SVC(),
                                    featuresTraining, labelsTraining, cv=5).mean()
                ScoreSVM.append(clfTestSVM)
                                    
                clfTestRF = cross_val_score(RFC(),
                                   featuresTraining, labelsTraining, cv=5).mean()
                ScoreRF.append(clfTestRF)
                
                clfTestNB = cross_val_score(NB(),
                               featuresTraining, labelsTraining, cv=5).mean()
                ScoreNB.append(clfTestNB)
        
            DTs = np.mean(ScoreDT)
            SVMs = np.mean(ScoreSVM)
            RFs = np.mean(ScoreRF)
            NBs = np.mean(ScoreNB)
                                         
            plt.scatter(x,DTs)
            plt.scatter(x,SVMs)
            plt.scatter(x,RFs)
            plt.scatter(x,NBs)
            plt.plot(x,DTs,'.b-')
            plt.plot(x,SVMs,'.y-')
            plt.plot(x,RFs,'.g-')
            plt.plot(x,NBs,'.r-')
     
        plt.title('Monte Carlo simulation of accuracy on non optimised classifiers')
        plt.xlim([5,405])
        blue_patch = mpatches.Patch(color='blue', label='Decision Trees')
        red_patch = mpatches.Patch(color='red', label='Naive Bayes')
        y_patch = mpatches.Patch(color='yellow', label='Support Vector Machine')
        g_patch = mpatches.Patch(color='green', label='Random Forest')
        plt.legend(handles=[g_patch,y_patch,blue_patch,red_patch],loc='lower right')
        plt.ylim([0.3,1.1])
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Samples')
        plt.grid()
        plt.show()