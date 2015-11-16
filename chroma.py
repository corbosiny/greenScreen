import numpy as np
import cv2

cap = cv2.VideoCapture(0)

low_blue = np.array([110,50,50])
up_blue = np.array([130,255,255])

cv2.namedWindow('sliders')

def nothing(value):
    pass


cv2.createTrackbar('H_U','sliders',0,255,nothing) # Value range
cv2.createTrackbar('H_L','sliders',0,255,nothing)
cv2.createTrackbar('S_U','sliders',0,255,nothing)
cv2.createTrackbar('S_L','sliders',0,255,nothing)
cv2.createTrackbar('V_U','sliders',0,255,nothing)
cv2.createTrackbar('V_L','sliders',0,255,nothing)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    bkgrnd = cv2.imread('wizard.jpg')

    hu = cv2.getTrackbarPos('H_U','sliders')
    hl = cv2.getTrackbarPos('H_L','sliders')
    su = cv2.getTrackbarPos('S_U','sliders')
    sl = cv2.getTrackbarPos('S_L','sliders')
    vu = cv2.getTrackbarPos('V_U','sliders')
    vl = cv2.getTrackbarPos('V_L','sliders')

    low_blue = np.array([hl,sl,vl])
    up_blue = np.array([hu,su,vu])

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, low_blue,up_blue)
    maskInv = (255-mask)
    output = cv2.bitwise_and(frame,frame,mask = maskInv)
    bkgrnd = cv2.bitwise_and(bkgrnd,bkgrnd,mask = mask)
    output = output + bkgrnd

    # Display the resulting frame
    cv2.imshow('orig',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('mask',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
