#This just standardizes the file names to simplify subsequent scripts. 

from pathlib import Path
import os

#These directories are not in the GitHub anywhere because of the quantity of data. 
download_dir = "/Users/daniel/NCSU-Mini/PSA/CV_Photo_Augment_Proj/opencv_synthetic_data/"
photo_dir = "/Users/daniel/NCSU-Mini/PSA/CV_Photo_Augment_Proj/input/"
photos = os.listdir(download_dir)
counter = 1
for photo in photos:
    os.rename(download_dir+photo,photo_dir+str(counter).zfill(5)+".jpg")
    counter+=1