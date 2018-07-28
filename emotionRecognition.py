import pickle
import cv2
from Extra.kateCV2 import face_detect,draw_rectangle,face_capture
import pandas as pd

font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade = cv2.CascadeClassifier('Extra/haarcascade_frontalface_default.xml')
pickle_in = open('EmotionRecognition/emotionRecognition.pickle', 'rb')
pickle_in2 = open('EmotionRecognition/pca.pickle', 'rb')
clf2 = pickle.load(pickle_in)
pca2 = pickle.load(pickle_in2)

data=pd.read_csv('EmotionRecognition/emotion_dict.csv',delimiter=',',header=None)
data.columns=['id','name']
emotion=list(data['name'])
#
def EmoRec(frame,f3,x,y,w,h):
    test = pca2.transform(f3.reshape(1, -1))
    #'''
    v = clf2.predict_proba(test)
    q = clf2.decision_function(test)
    #print("Confidence: " + str(q))
    #print("max confidence:" +str(max(q[0])))
    #print("Probability:"+str(v[0]))
    #print("\n")
    length=(len(v[0]))

    for i in range(length):
        if v[0][i] >.75 and q[0][i]==max(q[0]):
            if i==0:
                c=(0,0,255)
            elif i==1:
                c=(1, 216, 86)
            elif i==2:
                c=(255,255,255)
            elif i==3:
                c=(0,255,255)
            else:
                c=(255,255,255)
            draw_rectangle(frame,x,y,w,h,c)
            cv2.putText(frame, emotion[i], (x+w+10, y+h), font, 1.2, c, 2, cv2.LINE_AA)

def emotion_recognition(frame):
    detections = face_detect(frame, faceCascade=faceCascade)
    for x, y, w, h in detections:
        #c=(255,255,255)
        #draw_rectangle(frame, x, y, w, h,c)
        f3 = face_capture(frame, x, y, w, h)
        EmoRec(frame,f3,x,y,w,h)


def ERecognition():
    #print(emotion)
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        #frame=cv2.imread('Extra/test12.jpg')
        emotion_recognition(frame)
        # Display the resulting frame
        cv2.namedWindow('Video',)  # cv2.WINDOW_NORMAL
        cv2.imshow('Video', frame)

        if (cv2.waitKey(1) == ord('q')):
            break

    video_capture.release()
    cv2.destroyAllWindows()


#ERecognition()