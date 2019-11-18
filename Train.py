import numpy as np 
import argparse
from modelt import bulid_model
from dataset import Dataset,get_filenames
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from pyimagesearch.learning_rate_schedulers import PolynomialDecay
from keras.optimizers import SGD,Adam 
def parse_arguments():
    parser = argparse.ArgumentParser(description='Some parameters.')
    parser.add_argument(
        "--verbose",
        type=int,
        help="Image path",
        default=0
    ) 
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    img_shape = (256,256,3)
    model = bulid_model(img_shape)
    #model.summary()
    folder = '/home/zsh/Fast_scnn/'
    train_images, train_annotations, val_images, val_annotations = get_filenames(folder)
    batch_size = 8
    num_classes = 151
    image_size = [256,256]
    train_dataset = Dataset(
        image_size=image_size,
        image_filenames=train_images,
        annotation_filenames=train_annotations,
        num_classes=num_classes,
        batch_size=batch_size
        )

    val_dataset = Dataset(
        image_size=image_size,
        image_filenames=val_images,
        annotation_filenames=val_annotations,
        num_classes=num_classes,
        batch_size=batch_size
    )
    epochs = 400
    checkpoint = ModelCheckpoint(filepath='weights/weights-{epoch:03d}-{loss:.2f}.h5',monitor='loss',save_best_only=False,save_weights_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='loss',factor=0.5,cooldown=0,patience=5,min_lr=1e-7)
    #######################
    ## change learning rate schedule....
    optimizer = SGD(momentum=0.9,lr=0.045)
    #optimizer = SGD(momentum=0.9,lr=1e-3)
    
    schedule = PolynomialDecay(maxEpochs=epochs,initAlpha=0.045,power=0.9)
    lr_schedule = LearningRateScheduler(schedule)
    ###########################
    earlystopping = EarlyStopping(monitor='loss',patience=5,verbose=1)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    if args.verbose:
        try:
            model.load_weights('weights/weights-163-1.78.h5')
            print('successful load weights {}'.format('weights-163-1.78.h5'))
        except Exception as e:
            print('Error {}'.format(e))

    H = model.fit_generator(generator=train_dataset,validation_data=val_dataset,epochs=epochs,initial_epoch=163,callbacks=[checkpoint,lr_schedule])
