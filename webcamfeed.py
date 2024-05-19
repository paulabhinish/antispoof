import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True: 
    
    # Capture frame-by-frame 
    ret, frame1 = cap.read()
    print('Resolution: ' + str(frame1.shape[0]) + ' x ' + str(frame1.shape[1]))
    frame = frame1[0:480, 140:500]
    print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    # Show the captured image 
    cv2.imshow('WebCam', frame) 
      
    # wait for the key and come out of the loop 
    if cv2.waitKey(1) == ord('q'): 
        break
  
# Discussed below 
cap.release() 
cv2.destroyAllWindows() 