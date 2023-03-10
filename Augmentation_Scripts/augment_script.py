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



project_dir = "../../../../PSA/CV_Photo_Augment_Proj/"
input_dir = project_dir + "input/"
output_dir = project_dir +"test_run/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

files = [x for x in os.listdir(input_dir).sort() if ".jpg" in x]

for photo in files:
    photo = cv2.imread(input_dir+photo)
    photo = cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)
    scale = 0.7
    photo = downscale(photo,scale)
    dictionary_holder = split(photo,(1024,1024))

    separable = [blur,PCA,GaussNoise,ISONoise]

    for i in dictionary_holder:
        for j in range(0,5):
            holder = i #dictionary key is start of character code file name
            for k in range(0,random.randint(1,5)):
                [image,char] = separable[random.randint(0,(len(separable)-1))](dictionary_holder[i].astype(np.uint8))
                holder = holder + char
            cv2.imwrite(output_dir+photo+holder+".jpg",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))


