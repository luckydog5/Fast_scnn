import numpy as np 
import cv2
import colorsys
import random
from skimage.measure import find_contours
from skimage.draw import polygon
#from matplotlib.patches import Polygon
#from matplotlib import patches,lines

def apply_mask(image,mask,color,alpha=0.5,index=1):
    for c in range(3):
        image[:,:,c] = np.where(mask==index,image[:,:,c]*(1-alpha)+alpha*color[c]*255,image[:,:,c])
    return image


def random_colors(N,bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i/N,1,brightness)for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c),hsv))
    random.shuffle(colors)
    return colors

def display_instances(image,masks,class_ids,class_names=None,scores=None,title="",figsize=(20,8),ax=None,show_mask=True,show_bbox=True,colors=None,captions=None):

    colors = colors or random_colors(len(class_ids))
    masked_image = image.astype(np.uint32).copy()
    for i in range(len(class_ids)):
        masked_image = apply_mask(masked_image,masks[:,:,0],color)
