import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def collect_dataset():
    images=[]
    labels=[]
    labels_dic={}
    hand=[hand for hand in os.listdir("FingerCount/gestures/")]
    for i,hand in enumerate(hand):
        labels_dic[i]=hand
        for image in os.listdir("FingerCount/gestures/"+hand):
            if image.endswith('.jpg'):
                images.append(cv2.imread("FingerCount/gestures/"+hand+'/'+image,0))
                labels.append(i)
    return (images,np.array(labels),labels_dic)


def Htrain():
    images, labels, label_dic = collect_dataset()

    print(len(images))
    print(labels)
    print(label_dic)
    # print(images[0].shape)

    # plt.imshow(images[0])
    # plt.show()

    X_train = np.array(images)
    print(X_train.shape)
    train = X_train.reshape(len(X_train), -1)
    print(train.shape)
    # from sklearn.decomposition import PCA

    # pca1 = PCA()
    # pca1.fit(train)

    # plt.plot(np.cumsum(pca1.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative explained variance')
    # plt.show()
    # pca = PCA(n_components=1600)
    # new_train = pca.fit_transform(train)
    # print(new_train.shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    clf = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    with open('FingerCount/HandGestureClassification.pickle', 'wb') as f:
        pickle.dump(clf, f)
    f.close()

    # with open('Handpca.pickle', 'wb') as w:
    #    pickle.dump(pca, w)
    # w.close()



