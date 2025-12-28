import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = list()
    for pic_name in os.listdir(r"./" + data_path + r"/car/"):          #use r to un-escaped string, prevent \t, \n etc
        img = cv2.imread(r"./" + data_path + r"/car/" + pic_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (36, 16), interpolation = cv2.INTER_AREA)
        dataset.append((img,1))

    for pic_name in os.listdir(r"./" + data_path + r"/non-car/"):          #use r to un-escaped string, prevent \t, \n etc
        img = cv2.imread(r"./" + data_path + r"/non-car/" + pic_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (36, 16), interpolation = cv2.INTER_AREA)
        dataset.append((img,0))                         
    
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
