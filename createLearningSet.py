import numpy as np
from skimage.io import imread, imsave
import scipy.misc
import os
#import matplotlib.pyplot as plt
import cv2

features = []
labels = []

for root, dirs, files in os.walk('Img'):
    for name in files:
        if name.endswith('.png'):
            d = int(name[3:6])
            if (d<=11):
                print name
                label = d
            # lecture de l'image, mais seulement d'un des channels alpha
            # l'image originale fait 900x1200, on la rescale a 30x40
                img = 255 - imread(os.path.join(root, name))[:,:,0]
                for angle in np.arange(-10,10,0.5):
                    rotated_img = scipy.misc.imrotate(img,angle)
                    ctrs, hierarchy = cv2.findContours(rotated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    rect = cv2.boundingRect(ctrs[0])
                    center = (rect[0]+rect[2]//2,rect[1] + rect[3]//2)
                    half_width  = min(rotated_img.shape[1]-center[0],rect[2]//2)
                    half_height = min(rotated_img.shape[0]-center[1],rect[3]//2)
                    half_size = max(half_width,half_height)
                    cropped = np.zeros((2*half_size,2*half_size))
                    for i in range(-half_height,half_height):
                        for j in range(-half_width,half_width):
                            cropped[half_size+i,half_size+j] = rotated_img[center[1]+i,center[0]+j]
                    #cropped=img[y:y+height,x:x+width]
                    #plt.imshow(cropped)
                    #plt.show()

                    features.append(scipy.misc.imresize(cropped,(28,28)))
                    labels.append(label)

features = np.array(features)
labels = np.array(labels)

np.save('labels.npy',labels)
np.save('features.npy',features)