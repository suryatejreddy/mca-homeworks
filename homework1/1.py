
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import itertools
import json
import glob
import pickle
import tqdm
DIM = (100, 100)

FEATURE_PATH = "Correlogram_features/"
TRAINING_PATH = "train/"

def quantize_colour_space():
    all_colors = np.array(list(itertools.product(range(256), repeat=3))) #All possible colors of 256 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(np.float32(all_colors), 64, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #Peforming K Means to Reduce the entire space to a mere 64 colors
    label = label.reshape((256,256,256,1))
    center = np.uint8(center)
    return label, center

def process_image(img, labels, center):
    #Resizing to reduce compute
    img = cv2.resize(img, DIM, interpolation = cv2.INTER_AREA).reshape((-1,3))
    new_img = []
    for pixel in img:
        r = pixel[0]
        g = pixel[1]
        b = pixel[2]
        color_label = labels[r][g][b] #Get the label of the cluster center
        color_center = center[color_label][0].tolist() #Get color of the cluster center
        new_img.append(color_center)   
    new_img = np.array(new_img).reshape(DIM  + (3,)) #Reform image from cluster centers
    return new_img 

def get_surrounding_grid(x,y,k):
     return [ [(i,j) if  i >= 0 and i < DIM[0] and j >= 0 and j < DIM[1] else 0 for j in range(x-1-k, x+k)] for i in range(y-1-k, y+k)]

def get_neighbouring_cells(x,y,k):
    neighbours = []
    for dx in [-k,k]:
        for dy in [-k,k]:
            x_ = x + dx
            y_ = y + dy
            if x_ > - 1 and x_ < DIM[0] and y_ > - 1 and y_ < DIM[1]:
                neighbours.append([x_, y_])  
    return neighbours

def get_hash_value(pixel):
    r = str(pixel[0]).zfill(3)
    g = str(pixel[1]).zfill(3)
    b = str(pixel[2]).zfill(3)
    return r + ";" + g + ";" +  b

def create_hash_map(all_colors):
    d = {}
    for ind in range(len(all_colors)):
        hash_ = get_hash_value(all_colors[ind])
        d[hash_] = ind
    return d

def get_correlogram_feature(img, all_colors, K):
    feature = [] # Dimension of Colors x K
    color_index = create_hash_map(all_colors)
    for k in K:
        current_colors = [0 for i in range(len(all_colors))] # For Storing Count of Each Color That is Matched
        matched_count = 0
        
        for x in range(0, DIM[0], int(DIM[0]/10)): #Iterating Skipping 10 pixels at once
            for y in range(0, DIM[1], int(DIM[1]/10)):
                
                pixel = img[x][y]
                neighbours = get_neighbouring_cells(x,y,k)
                
                for cell in neighbours:
                    cell_x = cell[0]
                    cell_y = cell[1]
                    neighbour_pixel = img[cell_x][cell_y]
                    
                    if np.array_equal(pixel,neighbour_pixel): # The color is equal
                        hash_ = get_hash_value(pixel)
                        index = color_index[hash_]
                        current_colors[index] += 1 #Update Count
                        matched_count += 1
    
        if matched_count != 0: #Count Normalization
            normalized_colors = [i/matched_count for i in current_colors ]
        else:
            normalized_colors = current_colors[:]
        feature.append(normalized_colors)
    
    return feature


def create_feature_vectors():
    #Preprocessing
    print ("Quantizing Space")
    labels, colors = quantize_colour_space()
    print ("Completed Quantizing")
    K = [1,3,5,7]
    
    all_images = glob.glob("./images/*.jpg")
    
    for image_path in tqdm.tqdm(all_images, total = len(all_images)):
        name = image_path[image_path.rfind('/') + 1 : image_path.rfind(".")]
        image = cv2.imread(image_path)
        image = process_image(image, labels, colors)

        feature = get_correlogram_feature(image, colors, K)

        with open(FEATURE_PATH + name + ".json", 'w') as f:
            json.dump(feature, f)


# In[2]:


def get_similarity_value(vector_a, vector_b):
    #Each vector is of dimension 64x4
    similarity = 0
    num_k = len(vector_a)
    num_colors = len(vector_a[0])
    
    for k in range(num_k):
        for color in range(num_colors):
            pixel_a = vector_a[k][color]
            pixel_b = vector_b[k][color]
            numerator = abs(pixel_a - pixel_b)
            demoninator = 1 + pixel_a + pixel_b
            value = numerator / demoninator
            
            similarity += value
            
    similarity = similarity/num_colors
    return similarity


# In[3]:


import os
def load_all_vectors():
    index = {}
    all_files = os.listdir(FEATURE_PATH)
    for name in all_files:
        if ".json" in name:
            with open(FEATURE_PATH + name) as json_file:
                feature = json.load(json_file)
                index_name = name[:name.rfind('.')]
                index[index_name] = feature
    return index
index = load_all_vectors()


# In[4]:


def get_top_matches(base_image, k  = 15):
    base_feature = index[base_image]
    ranking = []
    for image, feature in index.items():
        if base_image != image:
            similarity = get_similarity_value(base_feature, feature)
            ranking.append([similarity, image])
    ranking = sorted(ranking, reverse = True)[:k]
    
    values = []
    for r in ranking:
        values.append(r[1])
        
    return values


# In[5]:


def get_query_image_from_file(filename):
    f = open(filename, "r")
    complete_line = f.read()
    f.close()
    
    first_word = complete_line.split(" ")[0]
    image_name = first_word[first_word.find('_') + 1 : ]
    
    return image_name

def get_query_results_from_file(query_name):
    ground_truth = glob.glob(TRAINING_PATH + "ground_truth/" + query_name + "*")
    split = {}
    split['all'] = []
    for file_ in ground_truth:
        category = file_[file_.rfind('_') + 1 : file_.rfind('.')]
        
        f = open(file_, "r")
        L = f.readlines()
        f.close()
        
        images = [i.strip() for i in L]
        
        split[category] = images
        split['all'].extend(images)
    
    return split
    
def load_ground_truth():
    query_path = TRAINING_PATH + "query"
    all_query_files = os.listdir(query_path)
    ground_truth = {}
    
    for query_file in all_query_files:
        query_name = query_file[:query_file.rfind('_')]
        
        query_image = get_query_image_from_file(TRAINING_PATH + "query/" + query_file)
        query_results_true = get_query_results_from_file(query_name)
        
        ground_truth[query_image] = query_results_true
        
    return ground_truth


# In[6]:


def get_single_stat(image, ground_truth, k):
    retrieved = set(get_top_matches(image, k))
    truth = set(ground_truth)
    common = retrieved.intersection(truth)
    
    precision = len(common)/len(retrieved)
    recall = len(common)/len(truth)
    if (precision + recall) > 0:
        f1_score =  2 * (precision * recall)/(precision + recall)
    else:
        f1_score = 0
    
    return precision, recall, f1_score


# In[7]:


def get_stats_for_image(image, ground_truth):
    precision = []
    recall = []
    f1_score = []
    
    for k in range(300,601):
        p, r, f = get_single_stat(image, ground_truth, k)
        precision.append(p)
        recall.append(r)
        f1_score.append(f)
    
    precision = sum(precision)/len(precision)
    recall = sum(recall)/len(recall)
    f1_score = sum(f1_score)/len(f1_score)
    
    return precision, recall, f1_score

def get_all_stats():
    ground_truth = load_ground_truth()
    
    precision = []
    recall = []
    f1_score = []
    
    for image, values in ground_truth.items():
        p, r, f = get_stats_for_image(image, values['all'])
        precision.append(p)
        recall.append(r)
        f1_score.append(f)
    
    print ("*" * 10)
    print ("Precision")
    print ("Max Value ", max(precision))
    print ("Min Value ", min(precision))
    print ("Avg Value ", sum(precision)/len(precision))
    print ("*" * 10)
    
    
    print ("*" * 10)
    print ("Recall")
    print ("Max Value ", max(recall))
    print ("Min Value ", min(recall))
    print ("Avg Value ", sum(recall)/len(recall))
    print ("*" * 10)
    
    
    print ("*" * 10)
    print ("F1 Score")
    print ("Max Value ", max(f1_score))
    print ("Min Value ", min(f1_score))
    print ("Avg Value ", sum(f1_score)/len(f1_score))
    print ("*" * 10)


# In[ ]:


get_all_stats()

