from keras.preprocessing import image
from modelt import bulid_model
import os
import glob
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import keras.backend as K 
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
img_h = 256
img_w = 256
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

def post_process(mask=None,N=20,min_size=8000):
    masks = []
    for i in range(N):
      temp = np.zeros((256,256))
      #temp = np.where(mask==(i+1),1,temp)
      p = np.where(mask==(i+1))
      print('{} has {} pixes'.format((i+1),len(p[0])))
      if len(p[0]) > min_size:
        temp[p] = 1

      masks.append(temp)
    return np.array(masks)
    
def load_model(inputs=(256,256,3),weights_path=None):
    img_shape = inputs
    model = bulid_model(img_shape)
    #model.summary()
    #weights_path = 'weights/weights-194-0.65.h5'
    model.load_weights(weights_path)
    return model 

def pre(model=None,image_path=None,image_gt=None):

    img_gt = image.load_img(image_gt,target_size=(img_h,img_w))
    img_gt = image.img_to_array(img_gt)

    img = image.load_img(image_path,target_size=(img_h,img_w))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x,axis=0)
    pred = model.predict(x)[0]
    pred = np.argmax(pred,2)
    return pred
def display_instances(image,masks,ax=None,show_mask=True):
    auto_show = False
    N = masks.shape[0]
    if not ax:
        _,ax = plt.subplots(1,figsize=(20,8))
        auto_show = True 
    colors = random_colors(N)
    height,width = image.shape[:2]
    ax.set_ylim(height + 10,-10)
    ax.set_xlim(-10,width + 10)
    ax.axis('off')
    ax.set_title("Results")
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        mask = masks[i]
        if show_mask:
            masked_image = apply_mask(masked_image,mask,color)
        padded_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),dtype=np.uint8)
        padded_mask[1:-1,1:-1] = mask 
        contours = find_contours(padded_mask,0.5)
        for verts in contours:
            verts = np.fliplr(verts) -1
            p = Polygon(verts,facecolor='none',edgecolor=color)
            ax.add_patch(p)
    masked_image = cv2.addWeighted(image,0.9,masked_image.astype(np.float32),0.1,0)
    ax.imshow(masked_image)
    if auto_show:
        plt.show()
if __name__ == '__main__':

    weights_path = 'weights/weights-194-0.65.h5'
    model = load_model(weights_path=weights_path)
    image_path = 'testImages/ADE_val_00000003.jpg'
    image_gt = os.path.splitext(image_path)[0] + '_seg.png'
    pred = pre(model,image_path,image_gt)
    print('pred shape is {}'.format(pred.shape))
    print(pred.shape[:-1])
    #print(img_gt)
    #pred[pred==0] = 255
    #pred = np.array(pred,dtype=np.uint8)
    print('unique pred : {}'.format(np.unique(pred)))
    t_img = image.load_img(image_path,target_size=(img_h,img_w))
    t_img = image.img_to_array(t_img)
    gt_ = image.load_img(image_gt,target_size=(img_h,img_w))
    gt_ = image.img_to_array(gt_)
    gt_ = np.where(gt_>20,0,gt_)
    print('unique mask: {}'.format(np.unique(gt_)))

    masks = post_process(pred)
    display_instances(t_img/255,masks)
    print('masks.shape: {}'.format(masks.shape))
    print('mask[0] shape : {}'.format(masks[0].shape))
    print('unique masks[0]: {}'.format(np.unique(masks[0])))
    """
    plt.figure(figsize=(20,8))
    plt.subplot(131)
    plt.imshow(pred)
    plt.subplot(132)
    plt.imshow(gt_*255)
    plt.subplot(133)
    plt.imshow(t_img/255)
    plt.show()
    plt.figure(figsize=(20,8))
    plt.subplot(131)
    plt.imshow(masks[0])
    plt.subplot(132)
    plt.imshow(masks[1])
    plt.subplot(133)
    plt.imshow(masks[2])
    plt.show()
    """