import numpy as np
import matplotlib.pyplot as plt
from GaussianNB import classify
from DecisionTree import classifyDT
from SVMcl import classify_SVM
from RandomForest import RFclassify
import matplotlib.patches as mpatches
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
import pandas as pd


def kfoldCV (X, y):
    kf = StratifiedKFold(y,n_folds=5)
    y = np.asarray(y)
    baba = []
    NB = []
    DT = []
    RF = []
    SVM = []
    #make training and testing datasets
    for train_index, test_index in kf:
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        AccuracyNB = classify(X_train,y_train,X_test,y_test)
        AccuracyDT = classifyDT(X_train,y_train,X_test,y_test)
        AccuracySVM = classify_SVM(X_train,y_train,X_test,y_test)
        AccuracyRF = RFclassify(X_train,y_train,X_test,y_test)
        NB.append(AccuracyNB)
        DT.append(AccuracyDT)
        RF.append(AccuracyRF)
        SVM.append(AccuracySVM)
    baba.append(NB)
    baba.append(DT)
    baba.append(RF)
    baba.append(SVM)
    df = pd.DataFrame(baba, index=['NB','DT', 'RF','SVM'])
    df.T.boxplot()
#    plt.subplots_adjust(bottom=0.25)
    plt.xticks(rotation=25)
    plt.ylim([0.6,1.05])
    plt.plot([0,0],[0,0],'r--')
    plt.title('(b)')
    plt.ylabel('Accuracy')
    plt.xlabel('Classifiers')
    plt.show()
        
def LeaveOneOut(X, y):
    loo = cross_validation.LeaveOneOut(n=len(y))
    y = np.asarray(y)
    Trues = []
    Falses = []
    #make training and testing datasets
    for train_index, test_index in loo:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Accuracy = RFclassify(X_train,y_train,X_test,y_test)
        if Accuracy == 1:
            Trues.append(Accuracy)
        else :
            Falses.append(Accuracy)
    Result = len(Trues)/(len(Trues)+len(Falses))
    print(Result)