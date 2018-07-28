import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def collect_dataset():
    images=[]
    labels=[]
    labels_dic={}
    people=[person for person in os.listdir("GenderClassification/trainingData/")]
    for i,person in enumerate(people):
        labels_dic[i]=person
        for image in os.listdir("GenderClassification/trainingData/"+person):
            if image.endswith('.jpg'):
                images.append(cv2.imread("GenderClassification/trainingData/"+person+'/'+image,0))
                labels.append(i)
    return (images,np.array(labels),labels_dic)

def Gtrain():
    images, labels, labels_dic = collect_dataset()
    # print (len(images))
    print(images[0])
    print (type(images))
    # print(labels_dic)
    X_train = np.array(images)
    print(X_train.shape)
    train = X_train.reshape(len(X_train), -1)
    print(train.shape)
    from sklearn.decomposition import PCA
    pca1 = PCA()
    pca1.fit(train)
    # plt.plot(np.cumsum(pca1.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative explained variance')
    # plt.show()
    pca1 = PCA(n_components=400)
    new_train = pca1.fit_transform(train)

    print(new_train.shape)
    #from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    #from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV,KFold
    from sklearn.model_selection import cross_val_score
    #models=[KNeighborsClassifier(),LogisticRegression(),SVC(kernel='linear',C=.001)]
    kf=KFold(n_splits=10,shuffle=True)
    #for model in models:
     #   result=cross_val_score(model,new_train,labels,cv=10,scoring='accuracy')
      #  print('model:%s'%model,np.mean(result))
    param_grid={'C':[.0001,.001,.01,1,10],'kernel':['linear','rbf'],'gamma':[.0001,.001,.01,.1,10]}
    kf=KFold(n_splits=10,shuffle=True)
    gs_svc=GridSearchCV(SVC(probability=True),param_grid=param_grid,cv=kf,scoring='accuracy')
    gs_svc.fit(new_train,labels)
    clf=gs_svc.best_estimator_
    #gaussian filtering, threshholding image, finding contours,cv2.dilate
    print(clf.score(new_train,labels))

    with open('GenderClassification/genderClassification.pickle','wb') as f:
        pickle.dump(clf,f)
    f.close()

    with open('GenderClassification/pca.pickle','wb') as w:
        pickle.dump(pca1,w)
    w.close()

