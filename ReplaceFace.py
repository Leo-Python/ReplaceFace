import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml") # The face cascade
smile_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_smile.xml')
cap = cv2.VideoCapture(1)
img = cv2.imread("img.png") # The image to put on your face. If you want to replace the image with another. Just take an image and drag it into the folder. Make sure to rename it to img.png. And remove the old img file

## The code

## The code works best if it isn't to bright or dark in your room. Try to keep it medium :)
while(True):
    
    ret, frame = cap.read()
    filtered = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)
    

    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = frame[y:y+h,x:x+w]
    cropFil = frame[y:y+h,x:x+w]
    





    for (x, y, w, h) in faces:
        XCenter = int(( (x+w) / 2))
        YCenter = int(( (y+h) / 2))
        print(XCenter, YCenter)
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 2) # draws a rectangle around your face. This only draws a rectangle on the frame
        roi_color = frame[y:y + h, x:x + w]
        width = roi_color.shape[1]
        height = roi_color.shape[0]



        dim = (width, height)
        resized = cv2.resize(img, dim,interpolation = cv2.INTER_AREA)
        print(x,y)
        filtered[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
        img = cv2.imread("img.png")
        for (sx, sy, sw, sh) in smiles:
            img = cv2.imread("smile.png") # The smile image to put on your mouth. If you want to replace the image with another. Just take an image and drag it into the folder. Make sure to rename it to smile.png. And remove the old smile.png file
            cv2.rectangle(frame, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) # draws a rectangle around your mouth whenever you smile. This only draws a rectangle on the frame


    cv2.rectangle(filtered,(0,60), (0,60), (255,0,0), 2)
    cv2.imshow("frame", filtered)
    #cv2.imshow("Background", crop) #Uncomment this if you want to see the rectangles around your face/mouth



# if you press "q" the program will break

    if cv2.waitKey(20) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
