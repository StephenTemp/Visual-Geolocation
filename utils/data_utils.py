# data_utils.py
# SUMMARY:

# IMPORTS
import os
import cv2
import random
import scipy.io
import numpy as np

from sklearn.cluster import KMeans
from typing import DefaultDict
import matplotlib.pyplot as plt
# END IMPORTS

# CONSTANTS
DATA_PATH = "../data/"
COORDS_FILE_PATH = "../data/GPS_Long_Lat_Compass.mat"

LABEL_DICT = {0 : "Orlando",
              1 : "Pittsburgh",
              2 : "New York"}

LOCAL_PARTITION_DICT = {0 : "East", 1 : "West"}
# END CONSTANTS

def plot_coords(coords, labels):
    lats, lons = np.hsplit(coords, 2)
    plt.scatter(lats, lons)
    plt.show()

class Macro_Classification:
    # init method or constructor   
    def __init__(self):
        self.data = None
        self.labels = None

        self.coords = scipy.io.loadmat(COORDS_FILE_PATH)

    def load_data(self, folder_paths=[], img_wd=100, img_ht=100):
        coords = self.coords

        data_X = list() # the instance attributes    
        coordinates = list()
    
        count = 0
        for folder in folder_paths:
            folder = DATA_PATH + folder
            for image in os.listdir(folder):
                if not image.endswith('.jpg'): continue
                image_name = image.split(".")[0]
                view = int(image_name[-1])
                
                coordinates.append(coords['GPS_Compass'][int(image_name[:-2]) - 1][:2])
                
                img_path = os.path.join(folder, image)
                img = cv2.imread( img_path, cv2.IMREAD_COLOR)
                
                img=cv2.resize(img, (img_ht, img_wd), interpolation = cv2.INTER_AREA)
                img = img
                img = img.astype('float32')
                img /= 255 
                data_X.append(img.T)
                count += 1

        self.data = (np.array(data_X), np.array(coordinates))
        return np.array(data_X), np.array(coordinates)

    def label_data(self):
        total_labels = []
        total_coords = self.data[1]
    
        lats, lons = np.hsplit(total_coords, 2)
        plt.scatter(lats, lons)
        
        for coords in total_coords:
            if(coords[0] < 34): total_labels.append(2)
            elif(coords[1] > -75): total_labels.append(0)
            elif(coords[1] < -75): total_labels.append(1)
            else: print("UNIDENTIFIED EXAMPLE")

        self.labels = np.array(total_labels)
        return np.array(total_labels)
    
    def sample_dataset(self, sample_num):
        dataset = self.data[0]
        labels = self.labels

        num_instances = dataset.shape[0]
        for i in range(sample_num):
            plt.subplot(1, sample_num, i + 1)
            rand_index = random.randint(0, num_instances)
            rand_image = dataset[rand_index]
            rand_class = labels[rand_index]
            image = np.moveaxis(rand_image, (0, 1, 2), (2, 1, 0))
            plt.imshow(image)
            plt.axis('off')
            plt.title(LABEL_DICT[rand_class])
            plt.gcf().set_size_inches(12, 5)
        plt.show()
    

class Panoramic_Classification:
    # init method or constructor   
    def __init__(self):
        self.data = None
        self.coordinates = None
        self.labels = None

        self.coords = scipy.io.loadmat(COORDS_FILE_PATH)
    
    def load_data(self, folder_paths=[], img_wd=100, img_ht=100):
        data_X = DefaultDict(lambda: []) # the instance attributes
        coordinates = {}
        
        for folder in folder_paths:
            folder = DATA_PATH + folder
            for image in os.listdir(folder):
                if not image.endswith('.jpg'): continue
                    
                image_name = image.split(".")[0]
                view = int(image_name[-1])
                
                img_path = os.path.join(folder, image)
                img = cv2.imread( img_path, cv2.IMREAD_COLOR)
                
                img=cv2.resize(img, (img_ht, img_wd), interpolation = cv2.INTER_AREA)
                img = img
                img = img.astype('float32')
                img /= 255 
                data_X[int(image_name[:-2])].append(img.T)
                coordinates[int(image_name[:-2])] = self.coords['GPS_Compass'][int(image_name[:-2]) - 1][:2]
        
        double_imgs = []
        double_labels = []
        for point in data_X:
            points = data_X[point]
            
            point_one = np.hstack((points[0], points[1]))
            point_two = np.hstack((points[2], points[3]))
            point_three = np.hstack((points[4], points[5]))
            
            double_imgs.append(point_one)
            double_labels.append(coordinates[point])
            
            double_imgs.append(point_two)
            double_labels.append(coordinates[point])
            
            double_imgs.append(point_three)
            double_labels.append(coordinates[point])
        
        self.data = np.array(double_imgs)
        print(self.data.shape)
        self.coordinates = np.array(double_labels)
        print(self.coordinates.shape)
        return self.data, self.coordinates 

    def label_data(self):
        total_labels = []
        total_coords = self.coordinates
        print(total_coords.shape)
        lats, lons = np.hsplit(total_coords, 2)
        plt.scatter(lats, lons)
        
        for coords in total_coords:
            if(coords[0] < 34): total_labels.append(2)
            elif(coords[1] > -75): total_labels.append(0)
            elif(coords[1] < -75): total_labels.append(1)
            else: print("UNIDENTIFIED EXAMPLE")

        self.labels = np.array(total_labels)
        return np.array(total_labels)
    
    def sample_dataset(self, sample_num):
        dataset = self.data
        labels = self.labels

        num_instances = dataset.shape[0]
        for i in range(sample_num):
            plt.subplot(1, sample_num, i + 1)
            rand_index = random.randint(0, num_instances)
            rand_image = dataset[rand_index]
            rand_class = labels[rand_index]
            image = np.moveaxis(rand_image, (0, 1, 2), (2, 1, 0))
            plt.imshow(image)
            plt.axis('off')
            plt.title(LABEL_DICT[rand_class])
            plt.gcf().set_size_inches(12, 5)
        plt.show()


class Local_Regression:
    # init method or constructor   
    def __init__(self):
        self.data = None
        self.coordinates = None
        self.labels = None
        self.coords = scipy.io.loadmat(COORDS_FILE_PATH)
    
    def load_data(self, folder_paths=[], img_wd=100, img_ht=100):
        coords = self.coords

        data_X = list() # the instance attributes    
        coordinates = list()
    
        count = 0
        for folder in folder_paths:
            folder = DATA_PATH + folder
            for image in os.listdir(folder):
                if not image.endswith('.jpg'): continue
                image_name = image.split(".")[0]
                
                coordinates.append(coords['GPS_Compass'][int(image_name[:-2]) - 1][:2])
                
                img_path = os.path.join(folder, image)
                img = cv2.imread( img_path, cv2.IMREAD_COLOR)
                
                img=cv2.resize(img, (img_ht, img_wd), interpolation = cv2.INTER_AREA)
                img = img
                img = img.astype('float32')
                img /= 255 
                data_X.append(img.T)
                count += 1

        self.data = np.array(data_X)
        self.labels = np.array(coordinates)
        return self.data, self.labels
    
    def sample_dataset(self, sample_num):
        dataset = self.data
        labels = self.labels

        num_instances = dataset.shape[0]
        for i in range(sample_num):
            plt.subplot(1, sample_num, i + 1)
            rand_index = random.randint(0, num_instances)
            rand_image = dataset[rand_index]
            rand_class = labels[rand_index]
            image = np.moveaxis(rand_image, (0, 1, 2), (2, 1, 0))
            plt.imshow(image)
            plt.axis('off')
            plt.title(rand_class)
            plt.gcf().set_size_inches(12, 5)
        plt.show()

    
class Local_Partition:
    # init method or constructor   
    def __init__(self):
        self.data = None
        self.labels = None
        self.coordinates = None

        self.coords = scipy.io.loadmat(COORDS_FILE_PATH)

    def load_data(self, folder_paths=[], img_wd=100, img_ht=100):
        coords = self.coords

        data_X = list() # the instance attributes    
        coordinates = list()
    
        count = 0
        for folder in folder_paths:
            folder = DATA_PATH + folder
            for image in os.listdir(folder):
                if not image.endswith('.jpg') or '(' in image: continue
                image_name = image.split(".")[0]
                
                coordinates.append(coords['GPS_Compass'][int(image_name[:-2]) - 1][:2])
                
                img_path = os.path.join(folder, image)
                img = cv2.imread( img_path, cv2.IMREAD_COLOR)
                
                img=cv2.resize(img, (img_ht, img_wd), interpolation = cv2.INTER_AREA)
                img = img
                img = img.astype('float32')
                img /= 255 
                data_X.append(img.T)
                count += 1

        self.data = np.array(data_X)
        self.coordinates = np.array(coordinates)
        return np.array(data_X), np.array(coordinates)

    def label_data(self):
        total_labels = []
        total_coords = self.coordinates

        plot_one = []
        plot_two = []
        for coords in total_coords:
            if(coords[1] > -79.9999): 
                total_labels.append(0)
                plot_one.append(coords)
            else: 
                total_labels.append(1)
                plot_two.append(coords)

        self.labels = np.array(total_labels)
        lats_one, lons_one = np.hsplit(np.array(plot_one), 2)
        lats_two, lons_two = np.hsplit(np.array(plot_two), 2)
        plt.scatter(lats_one, lons_one)
        plt.scatter(lats_two, lons_two)
        plt.show()
        return np.array(total_labels), (lats_one, lons_one), (lats_two, lons_two)
    
    def sample_dataset(self, sample_num):
        dataset = self.data
        labels = self.labels

        num_instances = dataset.shape[0]
        for i in range(sample_num):
            plt.subplot(1, sample_num, i + 1)
            rand_index = random.randint(0, num_instances)
            rand_image = dataset[rand_index]
            rand_class = labels[rand_index]
            image = np.moveaxis(rand_image, (0, 1, 2), (2, 1, 0))
            plt.imshow(image)
            plt.axis('off')
            plt.title(LOCAL_PARTITION_DICT[rand_class])
            plt.gcf().set_size_inches(12, 5)
        plt.show()


class Local_Cluster:
    # init method or constructor   
    def __init__(self):
        self.data = None
        self.labels = None
        self.coordinates = None

        self.coords = scipy.io.loadmat(COORDS_FILE_PATH)

    def load_data(self, folder_paths=[], img_wd=100, img_ht=100):
        coords = self.coords

        data_X = list() # the instance attributes    
        coordinates = list()
    
        count = 0
        for folder in folder_paths:
            folder = DATA_PATH + folder
            for image in os.listdir(folder):
                if not image.endswith('.jpg') or '(' in image: continue
                image_name = image.split(".")[0]
                
                coordinates.append(coords['GPS_Compass'][int(image_name[:-2]) - 1][:2])
                
                img_path = os.path.join(folder, image)
                img = cv2.imread( img_path, cv2.IMREAD_COLOR)
                
                img=cv2.resize(img, (img_ht, img_wd), interpolation = cv2.INTER_AREA)
                img = img
                img = img.astype('float32')
                img /= 255 
                data_X.append(img.T)
                count += 1

        self.data = np.array(data_X)
        self.coordinates = np.array(coordinates)
        return np.array(data_X), np.array(coordinates)

    def label_data(self):
        # CLUSTER POINTS
        kmeans = KMeans(
            init="random",
            n_clusters=2,
            n_init=10,
            max_iter=300,
            random_state=42)
        
        kmeans.fit(self.coordinates)
        clusters = kmeans.cluster_centers_
        
        plot_one = []
        plot_two = []
        
        labels = []
        for coord in self.coordinates:
            if( abs(np.sum(coord - clusters[0])) > abs(np.sum(coord - clusters[1]) ) ): 
                labels.append(0)
                plot_one.append(coord)
            else: 
                labels.append(1)
                plot_two.append(coord)
        
        lats_one, lons_one = np.hsplit(np.array(plot_one), 2)
        lats_two, lons_two = np.hsplit(np.array(plot_two), 2)
        plt.scatter(lats_one, lons_one)
        plt.scatter(lats_two, lons_two)
        plt.show()

        self.labels = np.array(labels)
        return self.labels 
    
    def sample_dataset(self, sample_num):
        dataset = self.data
        labels = self.labels

        num_instances = dataset.shape[0]
        for i in range(sample_num):
            plt.subplot(1, sample_num, i + 1)
            rand_index = random.randint(0, num_instances)
            rand_image = dataset[rand_index]
            rand_class = labels[rand_index]
            image = np.moveaxis(rand_image, (0, 1, 2), (2, 1, 0))
            plt.imshow(image)
            plt.axis('off')
            plt.title(LOCAL_PARTITION_DICT[rand_class])
            plt.gcf().set_size_inches(12, 5)
        plt.show()

