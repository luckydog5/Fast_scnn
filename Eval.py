from modelt import bulid_model
from dataset import Dataset,get_filenames
model = bulid_model()
model.summary()
folder = '/home/zsh/Fast_scnn/'
train_images, train_annotations, val_images, val_annotations = get_filenames(folder)
batch_size = 2
val_dataset = Dataset(
    image_size=[2048,1024],
    image_filenames=val_images,
    annotation_filenames=val_annotations,
    num_classes=21,
    batch_size=batch_size
)
try:
    model.load_weights('weights/weights-037-0.85.h5')
    print('successful load weights {}'.format('weights-037-0.85.h5'))
except Exception as e:
    print('Error {}'.format(e))
for i in range(10):
    f =  model.evaluate_generator(val_dataset,steps=10*(i+1))
    print('f is {}'.format(f))