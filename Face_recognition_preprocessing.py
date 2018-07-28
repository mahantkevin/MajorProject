import cv2
import os
from Extra.kateCV2 import draw_rectangle,face_capture,face_detect
import numpy as np
import pickle
import csv

# creating a dataset :
faceCascade = cv2.CascadeClassifier('Extra/haarcascade_frontalface_default.xml')

def face_save(folder,frame):
    detections = face_detect(frame, faceCascade=faceCascade)
    sample=(len(os.listdir(folder)))
    print(sample)
    for x, y, w, h in detections:
        c=(255,255,255)
        draw_rectangle(frame, x, y, w, h,c)
        f4 = face_capture(frame, x, y, w, h)
        cv2.imwrite(folder+'/'+str(sample)+'.jpg', f4)
        sample+=1

def add_face(name,frame):
    folder = "people/" + name.lower()
    if not os.path.exists(folder):
        os.mkdir(folder)
        face_save(folder,frame)
    else:
        face_save(folder,frame)

def face_data():
    name=input("name: ")
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        add_face(name, frame)
        sample = (len(os.listdir("people/" + name.lower())))
        #print('new sample number ', sample)
        if sample >400:
            break
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if (cv2.waitKey(1) == ord('q')):
            break
    video_capture.release()
    cv2.destroyAllWindows()

#creating a model:
def collect_dataset():
    images=[]
    labels=[]
    labels_dic={}
    path="FaceRecognition/people/"
    people=[person for person in os.listdir(path)]
    for i,person in enumerate(people):
        labels_dic[i]=person
        for image in os.listdir(path+person):
            if image.endswith('.jpg'):
                images.append(cv2.imread(path+person+'/'+image,0))
                labels.append(i)
    with open('FaceRecognition/name_dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in labels_dic.items():
            writer.writerow([key, value])

    return (images,np.array(labels),labels_dic)

def Ftrain():
    images, labels, labels_dic = collect_dataset()
    print(len(images))
    print(type(images))
    #print(labels_dic)
    X_train = np.array(images)
    print(X_train.shape)
    train = X_train.reshape(len(X_train), -1)
    print(train.shape)
    from sklearn.decomposition import PCA
    #pca1 = PCA()
    #pca1.fit(train)
    # plt.plot(np.cumsum(pca1.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative explained variance')
    # plt.show()
    pca1 = PCA(n_components=600)
    new_train = pca1.fit_transform(train)
    print(new_train.shape)
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    classifier=SVC(kernel='linear',probability=True)
    classifier.fit(new_train,labels)
    print(classifier.score(new_train,labels))
    with open('FaceRecognition/faceRecognition.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    f.close()

    with open('FaceRecognition/pca.pickle', 'wb') as w:
        pickle.dump(pca1, w)
    w.close()


#Ftrain()