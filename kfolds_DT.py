import numpy as np
import math as mt
import pandas as pd
import itertools
from scipy import stats
import matplotlib.pyplot as plt
from os import system
import graphviz as gv
from sklearn.tree import export_graphviz, DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]
        use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)
    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            this_xs = this_xs.reindex(np.random.permutation(this_xs.index))
        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)
        xs.append(x_)
        ys.append(y_)

    xs = pd.concat(xs)
    ys = pd.Series(data=np.concatenate(ys),name='target')
    return xs,ys
    
def small_regression(X,y):
    X = X.sample(60)
    index = X.index.tolist()
    label = []
    for n in index:
        label.append(y[n])
    X = X.reset_index()
    del X['index']
    X_train, y_train = X[:40] , label[:40]
    X_test, y_test = X[40:] , label[40:]
    regr_1 = DecisionTreeRegressor()
    regr_1.fit(X_train, y_train)

    
    y_1 = regr_1.predict(X_test)
    plt.figure()
    plt.scatter(X_train, y_train, c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="test", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()
    
class myownStratifiedKfolds():
    
    def __init__(self):
        pass
        
        
    def kfold_DT(self, X, y, N):
        
        labels = []
        folds = []
        feature = X.sample(len(X))
        indexfeatures = feature.index.tolist()
        indexfolds=np.array_split(indexfeatures,N)
        for k in indexfolds:
            folds.append(feature.loc[k])
            labels.append(y[k]) 
            
# Balanced classes sampling            
 """"    
        folds=[]
        labels = []
        xs,ys = balanced_subsample(X,y)
        l = int(len(ys)/N)
        u = int(len(ys)/2)
        ysp = ys[:u].tolist()
        xsp = xs[:u]
        ysn = ys[u:].tolist()
        xsn = xs[u:]
        for n in range(N):
            foldp = ysp[int((l/2)*n):int((l/2)*(n+1))]
            foldn = ysn[int((l/2)*n):int((l/2)*(n+1))]
            foldp.extend(foldn)
            foldxp = xsp[int((l/2)*n):int((l/2)*(n+1))]
            foldxn = xsn[int((l/2)*n):int((l/2)*(n+1))]
            df = foldxp.append([foldxn])
            labels.append(foldp)
            folds.append(df)"""

        fig = plt.figure()   
        print("***************** Start of kfold ******************")
        plt.subplot(111, axisbg='whitesmoke', frameon=True)

        for i in range(N):
            print("Investigation of fold %s"%(i+1))
            y_test,X_test = labels[i],folds[i]
            y_rest, X_rest = [],[]
            for j in range(N):
                if j != i:
                    y_rest.append(labels[j])
                    X_rest.append(folds[j])
            
            y_training, X_training = y_rest[1:], X_rest[1:]
            y_validation, X_validation = y_rest[0],X_rest[0]
            
            y_training = list(itertools.chain.from_iterable(y_training))
            X_training = pd.concat(X_training)
            
           # y_validation = list(itertools.chain.from_iterable(y_validation))
        #    X_validation = pd.concat(X_validation)
            
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_training, y_training)
           
            importancesAttributes = clf.feature_importances_
            indicesAttributes = np.argsort(importancesAttributes)[::-1]
            feature_names=['age', 'bp', 'sg','al',
                            'su','rbc','pc','pcc',
                            'ba','bgr','bu','sc',
                            'sod','pot','hemo','pcv',
                            'wbcc','rbcc','htn','dm',
                            'cad','appet','pe','ane','class'] 
            
            features = []
            for f in range(X_training.shape[1]):
                if importancesAttributes[indicesAttributes[f]] > 0 :
                    features.append(indicesAttributes[f])
                    
                    
            print('Validation accuracy before pruning : ')
            pred = clf.predict(X_validation)
            accuracyValidationbefore= accuracy_score(y_validation,pred)
            print(accuracyValidationbefore)
            X_trainingsmall = X_training.loc[:,features]
            clf.fit(X_trainingsmall,y_training) 
            
            print('Validation accuracy after pruning : ')
            X_validationsmall = X_validation.loc[:,features]
            pred2 = clf.predict(X_validationsmall)
            accuracyValidationafter=accuracy_score(y_validation,pred2)
            print(accuracyValidationafter)
            
            
            if accuracyValidationafter > accuracyValidationbefore:
                # Print the feature ranking
                print("Feature ranking:")

                for f in range(X_training.shape[1]):
                    print("%d. feature %d %s (%f)" % (f + 1, indicesAttributes[f],
                                                    feature_names[indicesAttributes[f]],
                                                    importancesAttributes[indicesAttributes[f]]))
                                                

                print('These are the results from fold %s'%i)
                print('the model is validated and pruned where the features are:')
                print(features)
                featurenames = []
                for n in features:
                    featurenames.append(feature_names[n])
                with open('tree.dot', 'w') as dotfile:
                    dot_data = export_graphviz(clf,
                        feature_names=featurenames,node_ids=True, rounded=True)
    
                system("dot -Tpng tree.dot -o tree.png")
                
                print('Test Accuracy:')
                pred3 = clf.predict(X_test.loc[:,features])
                accuracyTest = clf.score(X_test.loc[:,features], y_test)
                print(accuracyTest)
                
                
                
            plt.bar(i - 0.1, accuracyValidationbefore, width=.1, label='before selection',
                        color='navy')

            plt.bar(i, accuracyValidationafter,
                        width=.1, label='after selection', color='c')

            print("*************************************************")
        

        plt.title("Validation set accuracy before and after feature selection",y=1.05)
        plt.xlabel('Folds')
        plt.ylabel('Accuracy R')
        plt.xlim([-0.5,N])
        plt.ylim([0,1])
        plt.xticks(np.arange(0,N,1))
        plt.yticks(np.arange(0,1,0.2))
        plt.tick_params(direction='in')
        plt.legend(loc='center left',bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
        plt.grid(True)
        plt.show()
        
                                        

            
          
            
            
            
            
            
            
