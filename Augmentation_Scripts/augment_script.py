import cv2
import os
from pathlib import Path
import random
import albumentations as A
import numpy as np


def split(image,desired_shape):
    #Currently assumes that image shape is larger than desired_shape
    out = np.zeros((4,desired_shape[0],desired_shape[1],3))
    out[0] = image[0:desired_shape[0],0:desired_shape[1]]
    out[1] = image[0:desired_shape[0],-desired_shape[1]:]
    out[2] = image[-desired_shape[0]:,0:desired_shape[1]]
    out[3] = image[-desired_shape[0]:,-desired_shape[1]:]
    out = {
        "tl":out[0],
        "tr":out[1],
        "bl":out[2],
        "br":out[3]
    }
    return out

def downscale(image,scale):
    out = cv2.resize(image,(0,0),fx=scale,fy=scale)
    return out

def blur(image):
    out = [np.zeros(image.shape,dtype=np.uint8),""]
    process = A.GaussianBlur(p=1.0)
    out[0]=process(image=image)["image"]
    out[1]="g"
    return out

def PCA(image):
    out = [np.zeros(image.shape,dtype=np.uint8),""]
    process = A.FancyPCA(alpha=0.1,p=1.0)
    out[0]=process(image=image)["image"]
    out[1]="p"
    return out

def GaussNoise(image):
    out = [np.zeros(image.shape,dtype=np.uint8),""]
    process = A.GaussNoise(var_limit=(10.0,50.0),per_channel=True,p=1.0)
    out[0]=process(image=image)["image"]
    out[1]="n"
    return out

def ISONoise(image):
    out = [np.zeros(image.shape,dtype=np.uint8),""]
    process = A.ISONoise(color_shift=(0.01,0.1),intensity=(0.1,0.5),p=1.0)
    out[0]=process(image=image)["image"]
    out[1]="i"
    return out


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
project_dir = os.path.abspath("../../CV_Photo_Augment_Proj/")
input_dir =  os.path.abspath("../../CV_Photo_Augment_Proj/input/")
label_dir = os.path.abspath("../../CV_Photo_Augment_Proj/labels/")
run_name = "dataset2_with_labels"
output_im_dir =  os.path.abspath("../../CV_Photo_Augment_Proj/"+run_name+"/images/")
resized_labels = os.path.abspath("../../CV_Photo_Augment_Proj/"+run_name+"/labels/")
if not os.path.exists(output_im_dir):
    Path(output_im_dir).mkdir(parents=True)
if not os.path.exists(resized_labels):
    Path(resized_labels).mkdir(parents=True)

input_list = os.listdir(input_dir)
input_list.sort()
files = [x for x in input_list if ".jpg" in x]

for photo_num in files:
    print("Starting "+photo_num)
    photo = cv2.imread(input_dir+photo_num)
    photo_num = photo_num.split(".")[0]
    label = cv2.imread(label_dir)
    photo = cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)
    scale = 0.7
    non_separable = [downscale,split]
    for i in non_separable:
        photo = i(photo,scale)
        label = i(label,scale)
        scale = (1024,1024) 
        #Reusing this scale variable for image shape allows the loop to work.
        #It is just changing to specify the output image shape for the split
        #function after the scaling. 

    for i in label:
        #i is dictionary key, which is either "tl", "tr", "bl", or "br" for matching the augmented images. 
        image = label[i].astype(np.uint8)
        cv2.imwrite((resized_labels+photo_num+i+".jpg"),cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2BGR)) 

    separable = [blur,PCA,GaussNoise,ISONoise]

    for i in photo:
        for j in range(0,10):
            holder = i +"_" #dictionary key is start of character code file name
            image = photo[i].astype(np.uint8)
            for k in range(0,random.randint(1,7)):
                [image,char] = separable[random.randint(0,(len(separable)-1))](image)
                holder = holder + char   
            cv2.imwrite((output_im_dir+photo_num+holder+".jpg"),cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2BGR))


