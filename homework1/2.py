
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import itertools
from skimage.transform.integral import integral_image
from skimage.feature import hessian_matrix_det, peak_local_max
from skimage.filters import gaussian
import glob
import tqdm
import json

DIM = (100, 100)
FEATURE_PATH = "Sift_features/"
scale_values = [15 * (1.2/9), 27 * (1.2/9) , 39 * (1.2/9), 51 * (1.2/9)] 


# In[2]:


def get_surf_vector(image):
    #Preprocess Image
    feature = []
    
    image = cv2.resize(image, DIM, interpolation = cv2.INTER_AREA)
    gray_temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float')
    image = cv2.normalize(gray_temp, None, 0, 1, cv2.NORM_MINMAX)
    
    #Get Interal/Summation Representation
    image = integral_image(image)
    
    for scale in scale_values:
        #Get Hessian Determinant and Get Local Maximums
        det = gaussian(image, scale)
        blobs = peak_local_max(det)
        
        #We Need to Add the Scale Information to the feature
        for blob in blobs:
            x_coordinate = blob[0]
            y_coordinate = blob[1]
            point = [float(x_coordinate), float(y_coordinate), scale]
            feature.append(point)
            
    return feature

def create_feature_vectors():
    all_images = glob.glob("./images/*.jpg")
    
    for image_path in tqdm.tqdm(all_images, total = len(all_images)):
        name = image_path[image_path.rfind('/') + 1 : image_path.rfind(".")]
        image = cv2.imread(image_path)
        

        feature = get_surf_vector(image)

        with open(FEATURE_PATH + name + ".json", 'w') as f:
            json.dump(feature, f)


# In[3]:


create_feature_vectors()

