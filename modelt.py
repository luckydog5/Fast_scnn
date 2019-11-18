import tensorflow as tf
import keras 
import keras.backend as K 
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import DepthwiseConv2D
from keras.layers import SeparableConv2D
from keras.layers import Lambda 
from keras.layers import concatenate
from keras.layers import AveragePooling2D
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import add 
from keras.layers import Softmax 
from keras.optimizers import SGD,Adam 
from keras.models import Model 


def conv_block(inputs,conv_type,kernel,kernel_size,strides,padding='same',relu=True):
    ## custom conv layer
    ## may use separableconv
    if conv_type == 'ds':
        x = SeparableConv2D(kernel,kernel_size,padding=padding,strides=strides)(inputs)
    else:
        x = Conv2D(kernel,kernel_size,padding=padding,strides=strides)(inputs)
    x = BatchNormalization()(x)
    if relu:
        x = Activation('relu')(x)
    return x 
def _res_bottleneck(inputs,filters,kernel,t,s,r=False):
    ## residual block
    tchannel = K.int_shape(inputs)[-1]*t
    x = conv_block(inputs,'conv',tchannel,(1,1),strides=(1,1))
    x = DepthwiseConv2D(kernel,strides=(s,s),depth_multiplier=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = conv_block(x,'conv',filters,(1,1),strides=(1,1),padding='same',relu=False)
    if r:
        x = add([x,inputs])
    return x
def bottleneck_block(inputs,filters,kernel,t,strides,n):
    ## custom residual block.. real one...
    x = _res_bottleneck(inputs,filters,kernel,t,strides)
    for i in range(1,n):
        x = _res_bottleneck(x,filters,kernel,t,1,True)
    return x 
def pyramid_pooling_block(input_tensor,bin_sizes):
    ## use input_shape instead fixed shape.
    _,h,w,c = input_tensor.shape
    concat_list = [input_tensor]
    #w = 64
    #h = 32
    for bin_size in bin_sizes:
        #x = AveragePooling2D(pool_size=(w//bin_size,h//bin_size),strides=(w//bin_size,h//bin_size))(input_tensor)
        #x = Conv2D(128,(3,3),strides=(2,2),padding='same')(x)
        x = AveragePooling2D(pool_size=(h//bin_size,w//bin_size),strides=(h//bin_size,w//bin_size))(input_tensor)
        x = Conv2D(int(c) // len(bin_sizes),(3,3),strides=(2,2),padding='same')(x)
        #x = Lambda(lambda x: tf.image.resize_images(x,(w,h)))(x)
        x = Lambda(lambda x: tf.image.resize_images(x,(h,w)))(x)
        concat_list.append(x)
    return concatenate(concat_list)

def bulid_model(input_shape=(2048, 1024, 3)):
    input_layer = Input(shape=input_shape,name='input_layer')
    lds_layer = conv_block(input_layer,'conv',32,(3,3),strides=(2,2))
    lds_layer = conv_block(lds_layer,'ds',48,(3,3),strides=(2,2))
    lds_layer = conv_block(lds_layer,'ds',64,(3,3),strides=(2,2))
    gfe_layer = bottleneck_block(lds_layer,64,(3,3),t=6,strides=2,n=3)
    gfe_layer = bottleneck_block(gfe_layer,96,(3,3),t=6,strides=2,n=3)
    gfe_layer = bottleneck_block(gfe_layer,128,(3,3),t=6,strides=1,n=3)
    #gfe_layer = pyramid_pooling_block(gfe_layer,[2,4,6,8])
    gfe_layer = pyramid_pooling_block(gfe_layer,[1,2,3,6])
    ################################################################
    ## feature fushion #########
    ff_layer1 = conv_block(lds_layer,'conv',128,(1,1),padding='same',strides=(1,1),relu=False)
    ff_layer2 = UpSampling2D((4,4))(gfe_layer)
    ff_layer2 = DepthwiseConv2D((3,3),strides=(1,1),depth_multiplier=1,padding='same')(ff_layer2)
    ff_layer2 = BatchNormalization()(ff_layer2)
    ff_layer2 = Activation('relu')(ff_layer2)
    ff_layer2 = Conv2D(128,(1,1),strides=(1,1),padding='same',activation=None)(ff_layer2)
    ff_final = add([ff_layer1,ff_layer2])
    ff_final = BatchNormalization()(ff_final)
    ff_final = Activation('relu')(ff_final)
    ###############################################################
    ### Classifier ###
    classifier = SeparableConv2D(128,(3,3),padding='same',strides=(1,1),name='DSConv1_classifier')(ff_final)
    classifier = BatchNormalization()(classifier)
    classifier = Activation('relu')(classifier)

    classifier = SeparableConv2D(128,(3,3),padding='same',strides=(1,1),name='DSConv2_classifier')(classifier)
    classifier = BatchNormalization()(classifier)
    classifier = Activation('relu')(classifier)
    ## num classes.....
    classifier = conv_block(classifier,'conv',151,(1,1),strides=(1,1),padding='same',relu=False)
    classifier = Dropout(0.3)(classifier)
    classifier = UpSampling2D((8,8))(classifier)
    #classifier = Activation('softmax')(classifier)
    classifier = Softmax()(classifier)
    ############################################################
    fast_scnn = Model(inputs=input_layer,outputs=classifier,name='Fast_SCNN')
    #optimizer = SGD(momentum=0.9,lr=0.045)
    #optimizer = SGD(momentum=0.9,lr=1e-3)
    #fast_scnn.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return fast_scnn

if __name__ == '__main__':

    model = bulid_model()
    model.summary()
    layerss = model.layers 
    print(layerss[-1].output)