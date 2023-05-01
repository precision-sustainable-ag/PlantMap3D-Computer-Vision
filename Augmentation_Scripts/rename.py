#This just standardizes the file names to simplify subsequent scripts. 

from pathlib import Path
import os

#These directories are not in the GitHub anywhere because of the quantity of data. 
source_dir = "../../../psa_images/biomass-labebed-training/data/synthetic_images/Images/"
photo_dir = "../../../psa_images/biomass-labebed-training/data/synthetic_images/renamed/"
photos = os.listdir(source_dir)
counter = 1
for photo in photos:
    os.rename(source_dir+photo,photo_dir+str(counter).zfill(5)+".jpg")
    counter+=1
