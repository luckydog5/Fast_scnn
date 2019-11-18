from keras.preprocessing import image
from modelt import bulid_model
import os
import glob
import numpy as np 
import matplotlib.pyplot as plt
import cv2

img_shape = (256,256,3)
model = bulid_model(img_shape)
model.summary()
img_h = 256
img_w = 256
#img_h = 2048
#img_w = 1024

#image_path = 'ADEChallengeData2016/images/validation/ADE_val_00000731.jpg'
image_path = 'ADEChallengeData2016/images/training/ADE_train_00000693.jpg'
weights_path = 'weights/weights-163-1.78.h5'
gt_ = 'ADEChallengeData2016/annotations/training/ADE_train_00000693.png'
model.load_weights(weights_path)
img_gt = image.load_img(gt_,target_size=(img_h,img_w))
img_gt = image.img_to_array(img_gt)
#img_gt = img_gt[:,:,0]
#cv2.imshow('gt',img_gt)
img = image.load_img(image_path,target_size=(img_h,img_w))
x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x,axis=0)
pred = model.predict(x)[0]
pred = np.argmax(pred,2)
print('pred shape is {}'.format(pred.shape))
print(pred.shape[:-1])
#print(img_gt)
#pred[pred==0] = 255
pred = np.array(pred,dtype=np.uint8)

plt.imshow(pred)
plt.show()
