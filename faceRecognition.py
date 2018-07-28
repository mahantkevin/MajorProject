import pickle
import cv2
from Extra.kateCV2 import face_detect,draw_rectangle,face_capture
import pandas as pd

font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade = cv2.CascadeClassifier('Extra/haarcascade_frontalface_default.xml')
pickle_in = open('FaceRecognition/faceRecognition.pickle', 'rb')
pickle_in2 = open('FaceRecognition/pca.pickle', 'rb')
clf2 = pickle.load(pickle_in)
pca2 = pickle.load(pickle_in2)

data=pd.read_csv('FaceRecognition/name_dict.csv',delimiter=',',header=None)
data.columns=['id','name']
names=list(data['name'])
#
def faceRec(frame,f3,x,y,w,h):
    test = pca2.transform(f3.reshape(1, -1))
    v = clf2.predict_proba(test)
    q = clf2.decision_function(test)
    #print("Confidence: " + str(q))
    #print("max confidence:" +str(max(q[0])))
    #print("Probability:"+str(v[0]))
    #print("\n")
    length=(len(v[0]))
    for i in range(length):
        if v[0][i] >.75 and q[0][i]==max(q[0]):
            cv2.putText(frame, names[i], (x + w+10, y), font, 1.2, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,"Prob : "+str(round(v[0][i],2)),(x+w+10,y+h),font,0.7, (255, 255, 0), 1, cv2.LINE_AA)


def face_recognition(frame):
    detections = face_detect(frame, faceCascade=faceCascade)
    for x, y, w, h in detections:
        c=(255,255,0)
        draw_rectangle(frame, x, y, w, h,c)
        f3 = face_capture(frame, x, y, w, h)
        faceRec(frame,f3,x,y,w,h)


def FRecognition():
    print(names)
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read(0)
       # frame=cv2.imread('Extra/test14.jpg')
        face_recognition(frame)
        # Display the resulting frame
        #cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)

        if (cv2.waitKey(1) == ord('q')):
            break

    video_capture.release()
    cv2.destroyAllWindows()


#FRecognition()