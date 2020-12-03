import cv2
import glob
import os
import numpy as np
import pickle
import time
import imutils

img=cv2.imread("11.png")
revert = imutils.resize(img, width=1100)
start_time = time.time()
gray = cv2.cvtColor(revert, cv2.COLOR_BGR2GRAY) 
J = cv2.integral(gray)
ret, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
image_final = cv2.bitwise_and(gray,gray, mask=mask)
ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY_INV)  # for black text , cv2.THRESH_BINARY_INV
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,1))
 # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
list=[]
  
for contour in contours:
       # get rectangle bounding contour
       [x, y, w, h] = cv2.boundingRect(contour)
       if w+x!=gray.shape[1] and y+h!=gray.shape[0] and w>9 and h>9:
           list.append([x,y,w+x,y+h])
           bb=np.array(list)
           cv2.rectangle(revert, (x, y), (x + w, y + h), (0, 255, 255), 1)
       continue
  #print(bb.shape)
  #cv2.imshow("lines",cv_img) 
st=time.time()

 #cv2.imwrite(filename,cv_img)


length=bb.shape[0]

m=np.zeros(shape=(bb.shape[0],12))
n=np.zeros(bb.shape[0])
for l in range (bb.shape[0]):           
   for i in range(bb[l,1],bb[l,3]-3,4):  
    for j in range(bb[l,0],bb[l,2]-4,4):
      n[l]=n[l]+1

      white=J[i,j]+J[i+1,j+3]-J[i,j+3]-J[i+1,j]         #1h maska
      black=J[i+1,j]+J[i+3,j+3]-J[i+1,j+3]-J[i+3,j]
      if (white/3-black/6)>30:
        m[l,0]=m[l,0]+1

      black=J[i+2,j]+J[i+3,j+3]-J[i+3,j]-J[i+2,j+3]  #2h maska
      white=J[i,j]+J[i+2,j+3]-J[i+2,j]-J[i,j+3]
      if (white/6-black/3)>30:
        m[l,1]=m[l,1]+1




      black=J[i,j]+J[i+1,j+3]-J[i,j+3]-J[i+1,j]   #4h maska
      white=J[i+1,j]+J[i+3,j+3]-J[i+1,j+3]-J[i+3,j]
      if (white/6-black/3)>30:
        m[l,2]=m[l,2]+1


      white=J[i+1,j+1]+J[i+2,j+2]-J[i+1,j+2]-J[i+2,j+1]    #10h maska 3x3
      black=J[i,j]+J[i+3,j+3]-J[i+3,j]-J[i,j+3]-white
      if (white-black/8>30):
        m[l,3]=m[l,3]+1

      black=J[i,j]+J[i+3,j+1]-J[i,j+1]-J[i+3,j]       #11h maska
      white=J[i,j+1]+J[i+3,j+3]-J[i+3,j+1]-J[i,j+3]
      if (white/6-black/3)>30:
        m[l,4]=m[l,4]+1

      black=J[i,j+2]+J[i+3,j+3]-J[i,j+3]-J[i+3,j+2]    #12h maska
      white=J[i,j]+J[i+3,j+2]-J[i,j+2]-J[i+3,j]
      if (white/6-black/3)>30:
        m[l,5]=m[l,5]+1

      black=J[i+3,j+2]+J[i,j+1]-J[i+3,j+1]-J[i,j+2] #13h maska
      white=J[i,j]+J[i+3,j+3]-J[i+3,j]-J[i,j+3]-black
      if (white/6-black/3)>30:
        m[l,6]=m[l,6]+1

      black=J[i+2,j+3]+J[i+1,j]-J[i+1,j+3]-J[i+2,j]          #haar_b
      white=J[i,j]+J[i+3,j+3]-J[i+3,j]-J[i,j+3]-black
      if (white/6-black/3)>30:
        m[l,7]=m[l,7]+1

      black=J[i+2,j]+J[i+4,j+2]-J[i+2,j+2]-J[i+4,j]      #haar_a   2x4
      white=J[i,j]+J[i+2,j+2]-J[i,j+2]-J[i+2,j]
      if (white/4-black/4)>30:
         m[l,8]=m[l,8]+1

      black=J[i,j+1]+J[i+4,j+2]-J[i,j+2]-J[i+4,j+1]    #haar_c   4x2
      white=J[i,j]+J[i+4,j+1]-J[i+4,j]-J[i,j+1]
      if (white/4-black/4)>30:
        m[l,9]=m[l,9]+1


      black=J[i,j+1]+J[i+2,j+2]-J[i,j+2]-J[i+2,j+1]      #haar_d 2x3
      white=J[i+2,j+3]+J[i,j]-J[i+2,j]-J[i,j+3]-black
      if (white/4-black/2)>30:
         m[l,10]=m[l,10]+1


      black1=J[i+2,j+4]+J[i,j+2]-J[i,j+4]-J[i+2,j+2]   #HAAR_E
      black2=J[i+4,j+2]+J[i+2,j]-J[i+2,j+2]-J[i+4,j]
      black=black1+black2
      white=J[i+4,j+4]+J[i,j]-J[i,j+4]-J[i+4,j]-black
      if (white/8-black/8)>30:
         m[l,11]=m[l,11]+1
full=np.array(m/n[:,None],dtype=np.float64)

# for l in range (bb.shape[0]):
 #    print("[",full[l,0],",",full[l,1],",",full[l,2],",",full[l,3],",",full[l,4],",",full[l,5],",",full[l,6],",",full[l,7],",",full[l,8],",",full[l,9],",",full[l,10],",",full[l,11],"]",",") 

pickle_in=open("@12_mask_1500.pickle","rb")     #fortwnw to training model
loaded_model=pickle.load(pickle_in)


new=[]
for i in range (length):          #proetoimasi agia na mpoune sto som network
    b=np.array(full[i])
    b=b.reshape(1,12)
    new.append(b)

new_labels=loaded_model.predict(new)
 #print("new labels",new_labels)


new_label=np.array(new_labels)
unique_element, counts_element = np.unique(new_label, return_counts=True)
 #print("Frequency of unique values of the said array:")
net=np.asarray((unique_element, counts_element))            #poses fores energopoieitai o ka8e neyrwnas sto tyxaio paradeigma
print("η τοπολογία του νετ ειναι",net)
print("--- %s seconds ---" % (time.time() - start_time))

n0=[]
n1=[]

for i ,label in enumerate(new_labels):
  if   label==3   :
    n0.append(i)
  if label==2 or label==0 or label==1: #txt neuron:
      n1.append(i)
st=time.time()
#img_0=cv2.imread("cv_img")
#revert0 = imutils.resize(cv_img, width=1100)
#img_1=cv2.imread("cv_img")
#revert1 = imutils.resize(cv_img, width=1100)



 #st=st+1   
  #filename="%s_txt.jpg"%st
  #cv2.imwrite(filename,revert0)
if len(n1)>0:
  for i in  (n1):
    bb_1=np.array(bb[i])
    x1=bb_1[0]
    y1=bb_1[1]
    x2=bb_1[2]
    y2=bb_1[3]
    cv2.rectangle(revert, (bb_1[0],bb_1[1]), (bb_1[2],bb_1[3] ),(255, 0, 255), 3)
    
  filename="%s_txt.jpg"%st
  cv2.imshow("final",revert)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imwrite(filename,revert)
