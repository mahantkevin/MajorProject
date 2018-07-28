import pickle
import cv2
from Extra.kateCV2 import face_detect,draw_rectangle,face_capture


font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade = cv2.CascadeClassifier('Extra/haarcascade_frontalface_default.xml')
#eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
pickle_in = open('GenderClassification/genderClassification.pickle', 'rb')
pickle_in2 = open('GenderClassification/pca.pickle', 'rb')
clf = pickle.load(pickle_in)
pca = pickle.load(pickle_in2)
label = {0: 'Female', 1: 'Male'}


def genderClass(frame,f3,x,y,w,h):
    test = pca.transform(f3.reshape(1, -1))
    z = clf.predict_proba(test)
    if z[0][0] > 0.80:
        # color is in BGR
        c=(180, 105, 255)
        cv2.putText(frame, label[0], (x + w+10, y + h), font, 0.8,c, 2, cv2.LINE_AA)
        draw_rectangle(frame, x, y, w, h, c)

    elif z[0][1] > 0.80:
        c=(255, 103, 25)
        cv2.putText(frame, label[1], (x + w+10, y + h), font, 0.8,c, 2, cv2.LINE_AA)
        draw_rectangle(frame,x,y,w,h,c)

    else:
        cv2.putText(frame, "...", (x + w, y + h), font, 1.2, (217, 243, 255), 2, cv2.LINE_AA)


def gender_classification(frame):
    detections = face_detect(frame, faceCascade=faceCascade)
    for x, y, w, h in detections:
        #draw_rectangle(frame, x, y, w, h)
        f3 = face_capture(frame, x, y, w, h)
        genderClass(frame,f3,x,y,w,h)


def Gclassification():
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
       # frame=cv2.imread('Extra/test8.jpg')
        gender_classification(frame)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if (cv2.waitKey(1) == ord('q')):
           break
    video_capture.release()
    cv2.destroyAllWindows()


#Gclassification()