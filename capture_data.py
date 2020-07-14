'''This script is to take pictures with the press of spacebar, given the label and number of pictures required'''

import numpy as np
import cv2 


cnt = 0 #setting count to 0

cap = cv2.VideoCapture(0)

label = input('label: ')
max_amt= input('max_mat: ')
file_path = input('path: ')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if return value is false, we skip the block
    if not ret:
        continue
    
    # Our operations on the frame come here 
    # insert rectangle box
    cv2.rectangle(frame,(60,60),(300,300),(255,0,0),2)

    #setting up our region of interest from where we want the pictures to be taken
    roi = frame[60:300, 60:300]
    #flip the image
    frame = cv2.flip(frame,1)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    
    # if space is clicked 
    if cv2.waitKey(1) & 0xFF==ord(' '):
        #while True:
            
        # we will save each image with a certain code name
        filename = str(file_path)+"\\{}{}.jpg".format(label,cnt)
        cv2.imwrite(filename,roi)
        
         #code to output and show the details 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Collecting {}".format(cnt),(5, 50), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Collecting images", frame)
        cnt += 1 #increment the counter

    # break if max amount reached
    if cnt==int(max_amt):
        break
        
        
    # press esc to end the capture
    k =  cv2.waitKey(1) & 0xFF 
    if k==27:
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
