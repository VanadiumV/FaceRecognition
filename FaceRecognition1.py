#Read a VIDEO FROM Web Cam using OpenCV
#Face Detectionin Video
#Click 20 PICTURES OF THE person who comes in front of camera

import cv2
import numpy as np
#Create a Camera Object
cam=cv2.VideoCapture(0)

#Ask the anme
fileName=input("Enter the name of the person:")
datset_path="./data/"   #folders accordingly for diff person
offset = 20

#Model
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Create a list of save face data
faceData = []
skip = 0


#Read image from Camera Object
while True:
    success,img=cam.read()
    if not success:
        print("Reading Camera Failed!")
        
#Store the gray images
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #bgr->rgb CONVERTING IMG TO GRAYSCALE PARAMETER

    faces=model.detectMultiScale(img,1.3,5) #pass img 2 detector
    #pick the face with the largest bounding box(to avoid FPs in image)
    faces=sorted(faces,key=lambda f:f[2]*f[3]) #sorting all faces acc to the largest face that is available
    #pick the largest face
    if len(faces)>0:    ##
        f = faces[-1]

    # for f in faces[-1:]:    #largest face tht is sorted   ##add an offset part of face your cropping off
        x,y,w,h=f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        #crop and save the largest face (2darr->{y,x})
        cropped_face = img[y - offset:y+h + offset,x - offset:x + offset +w]

        cropped_face=cv2.resize(cropped_face,(100,100))
        skip += 1
        
        if skip % 10 == 0:  #after every 10 faces ill save 1 face
           faceData.append(cropped_face)
        print("Saved so far " + str(len(faceData))) #so many faces have been saved so far

    cv2.imshow("Image Window",img)
    #cv2.imshow("Cropped Face",cropped_face)
    
    key = cv2.waitKey(1)  #Pause here for 1ms before you read the next image
    if key == ord('q'):
        break
    
#Write the faceData on the disk so that we can reuse them later
faceData = np.asarray(faceData)
m = faceData.shape[0]   ## 
faceData = faceData.reshape((m,-1)) ##side poses too for this

print(faceData.shape)

##Save on the Disk as file    
filepath = dataset_path + fileName + ".npy"
np.save(filepath,faceData)
print("Data Saved Successfully" + filepath)

#Release Camera,and Destroy Window
cam.release()
cv2.destroyAllWindows()


#you need to flatten the data in training data