import cv2
import numpy as np
import math
import pickle
from Extra.kateCV2 import imageThresh

pickle_in = open('FIngerCount/HandGestureClassification.pickle', 'rb')
clf = pickle.load(pickle_in)

def hand_recognition(img):
    cv2.rectangle(img, (0, 200), (200, 400), (0, 255, 0), 0)
    crop_img = img[200:400, 0:200]

    thresh = imageThresh(crop_img)

    # show thresholded image
    cv2.imshow('Thresh', thresh)

    image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        # apply cosine rule here
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
# ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
        cv2.line(crop_img, start, end, [0, 255, 0], 2)

    if count_defects == 1:
        cv2.putText(img,"Number of Fingers: 2",(200,50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (26, 58, 127), 2, cv2.LINE_AA)
    elif count_defects == 2:
        cv2.putText(img,"Number of Fingers: 3",(200,50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (26, 58, 127), 2, cv2.LINE_AA)
    elif count_defects == 3:
        cv2.putText(img,"Number of Fingers: 4",(200,50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (26, 58, 127), 2, cv2.LINE_AA)
    elif count_defects == 4:
        cv2.putText(img,"Number of Fingers: 5",(200,50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (26, 58, 127), 2, cv2.LINE_AA)
    else:
        r = (clf.predict(thresh.reshape(1, -1)))[0]
        if r==0:
            cv2.putText(img, "Number of Fingers: 0", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (26, 58, 127), 2,
                        cv2.LINE_AA)
        elif r==1:
            cv2.putText(img, "Number of Fingers: 1", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (26, 58, 127), 2,
                        cv2.LINE_AA)

    # show appropriate images in windows
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

def Hrecogniton():
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        hand_recognition(frame)
        # Display the resulting frame
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)

        if (cv2.waitKey(1) == ord('q')):
            break
    video_capture.release()
    cv2.destroyAllWindows()

