import cv2
import os
import numpy as np
import pickle
import csv

# creating a dataset :
faceCascade = cv2.CascadeClassifier('Extra/haarcascade_frontalface_default.xml')

#creating a model:
def collect_dataset(flag):
    images=[]
    labels=[]
    labels_dic={}
    if flag==1:
        path="Extra/KDEF_images/"
    elif flag==2:
        path="Extra/pro_images/"

    people=[person for person in os.listdir(path)]
    for i,emotion in enumerate(people):
        labels_dic[i]=emotion
        for image in os.listdir(path+emotion):
            if image.endswith('.jpg') or image.endswith('.JPG'):
                images.append(cv2.imread(path+emotion+'/'+image,0))
                labels.append(i)
    with open('EmotionRecognition/emotion_dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in labels_dic.items():
            writer.writerow([key, value])

    return (images,np.array(labels),labels_dic)

def Etrain(flag):
    images, labels, labels_dic = collect_dataset(flag)
    print(len(images))
    print(type(images))
    print(labels_dic)
    X = np.array(images)
    print(X.shape)
    train = X.reshape(len(X), -1)
    print(train.shape)
    from sklearn.decomposition import PCA
    pca1 = PCA()
    pca1.fit(train)

    pca1 = PCA(n_components=400)
    new_train = pca1.fit_transform(train)
    print(new_train.shape)
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(new_train,labels,test_size=0.01)


    from sklearn.svm import SVC
    #classifier=SVC(kernel='poly',gamma=10,C=100,degree=3)
    classifier = SVC(kernel='linear',probability=True)
    #classifier.fit(new_train,labels)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_train,y_train))
    print(classifier.score(X_test,y_test))

    with open('EmotionRecognition/emotionRecognition.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    f.close()

    with open('EmotionRecognition/pca.pickle', 'wb') as w:
        pickle.dump(pca1, w)
    w.close()
