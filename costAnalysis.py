from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def cost_analysis(X,y):
    X = X.sample(400)
    index = X.index.tolist()
    label = []
    for n in index:
        label.append(y[n])
    
    X= X.reset_index()
    del X['index']
    X_train, y_train = X[:200] , label[:200]
    X_test, y_test = X[200:] , label[200:]
    
    featuresranking = [14,2,3,12,10,6,16,17,15,11,8,1,4,5,7,23,9,22,13,18,19,20,21,0]
    feature_names=['hemo', 'sg', 'al','sod',
                    'bu','pc','wbcc','rbcc',
                    'pcv','sc','ba','bp',
                    'su','rbc','pcc','ane',
                    'bgr','pe','pot','htn',
                    'dm','cad','appet','age']
    cost = [1.65,0,25,3.2,11.85,30,30,30,1.62,14,50,0,20,39,27.64,20,0,49,0,18.4,50,0,0]
    featurestofit = []
    x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    error = []
    plt.xticks(x,feature_names)
    plt.xticks(range(24), feature_names, rotation=45)
    for i,f in enumerate(featuresranking):
        featurestofit.append(f)
        clf = DecisionTreeClassifier()
        clf.fit(X_train.loc[:,featurestofit], y_train)
        predict = clf.score(X_test.loc[:,featurestofit], y_test)
        print(X_test.loc[:,featurestofit])
        error.append(predict)
        plt.plot(x[i], predict,'o') 
    
    price=0
    for i,txt in enumerate(cost):
        price = price + cost[i]
        plt.annotate(round(price,0),(x[i],error[i]))
        
    plt.plot(x,error,'g-')
    plt.title("Change of Accuracy and cost (USD) of CKD detection with increase of predictive attributes")
    plt.xlabel('Features')
    plt.ylabel('Accuracy R')
    plt.xlim([-0.5,24])
    plt.ylim([0.8,1.0])
    plt.show()
    