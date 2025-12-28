import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    fileIn = open('./data/detect/detectData.txt', 'r', encoding='utf-8')
    fileOut = open('./data/detect/.ML_Models_pred.txt', 'w', encoding='utf-8')
    detected = fileIn.read()
    cord = re.split(r'[\s]', detected)
    cord.pop(len(cord)-1)
    cord = list(map(int, cord))
    toatalSpace = cord.pop(0)

    cap = cv2.VideoCapture(r'./data/detect/video.gif')
    while True:
        ret, frame = cap.read()
        if ret:
            for count in range(toatalSpace):
                imgCrop = crop(cord[count * 8], cord[count * 8 + 1], cord[count * 8 + 2], cord[count * 8 + 3], cord[count * 8 + 4],
                                cord[count * 8 + 5], cord[count * 8 + 6], cord[count * 8 + 7], frame)
                imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgCrop = cv2.resize(imgCrop, (36, 16), interpolation=cv2.INTER_AREA)
                imgCrop = imgCrop.reshape(1, 36 * 16)
                if clf.classify(imgCrop):
                    spaceOccuCord = np.array([cord[count * 8], cord[count * 8 + 1], cord[count * 8 + 2], cord[count * 8 + 3], cord[count * 8 + 6],
                                               cord[count * 8 + 7], cord[count * 8 + 4], cord[count * 8 + 5]])
                    spaceOccuCord = spaceOccuCord.reshape((-1, 1, 2))
                    cv2.polylines(frame, [spaceOccuCord], True, (0, 255, 0), 2)
                fileOut.write(str(clf.classify(imgCrop)))
                if count != toatalSpace-1:
                    fileOut.write(" ")
            fileOut.write("\n")
            cv2.imshow('detect', frame)
            key = cv2.waitKey(500)
            if key == 27:
              break
            
        else:
            break
            
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
