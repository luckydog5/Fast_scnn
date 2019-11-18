from PIL import Image
import numpy as np 
import keras 
from glob import glob
def get_filenames(folder):
    """This function get image and annotation file names from a dataset.

    Args:
        folder: Folder with ADE20K dataset. This directory should
            contain a folder "ADEChallengeData2016"

    Returns:
        train_images: List containing training images from dataset
        train_annotations: List containing segmented mask for training images
        val_images: List containing validation images from dataset
        val_annotations: List containing segmented mask for validation
            images
    """
    train_images = sorted(
        glob(f"{folder}ADEChallengeData2016/images/training/*"))
    train_annotations = sorted(glob(
        f"{folder}ADEChallengeData2016/annotations/training/*"))
    val_images = sorted(
        glob(f"{folder}ADEChallengeData2016/images/validation/*"))
    val_annotations = sorted(
        glob(f"{folder}ADEChallengeData2016/annotations/validation/*"))
    return train_images, train_annotations, val_images, val_annotations

class Dataset(keras.utils.Sequence):
    """Dataset class implemented by Keras API
    
    Attributes:
        image_size: List containing image size.
            ist structure: (image_height,image_width)
        image_filenames: List containing all image file names.
        annotation_filenames: List containing all annotation file names for each iamge
        num_classes: Number of classes.
        batch_size: Dataset batch size

    """
    def __init__(self,image_size,image_filenames,annotation_filenames,num_classes,batch_size):
        self.image_size = image_size
        self.num_classes = num_classes
        self.image_filenames = image_filenames
        self.annotation_filenames = annotation_filenames
        self.batch_size = batch_size
        self.dataset = self.get_file_path()

    def get_file_path(self):
        """Gain path of each image im the dataset.

        Returns: List containing images and annotations,divided into batches. 
            List strcture:
                [
                    #first batch
                    [
                        [image_filename1,image_filename2,...],[annotation_filename1,annotation_filename2,...]
                    ]

                    # second batch
                    [
                        [image_filename...],[annotation_filename.....]
                    ]
                ]

        """
        dataset = []
        for i in range(len(self.image_filenames)//self.batch_size):
            dataset.append([self.image_filenames[i*self.batch_size:(i+1)*self.batch_size],self.annotation_filenames[i*self.batch_size:(i+1)*self.batch_size]])

        if len(self.image_filenames) % self.batch_size !=0:
            idx = len(self.image_filenames) // self.batch_size
            dataset.append([self.image_filenames[idx*self.batch_size:],self.annotation_filenames[idx*self.batch_size:]])
        return dataset 

    def annotation_processing(self,annotations):

        """This function changes the mask of images, turning each pixel into one-hot vector.
        Params:
            annotations: Numpy array containing  images. Array shape: (num_images,img_height,img_width)

        Returns:
            vectorized_annotation: Numpy array with shape (num_images,img_height,img_width,num_classes)
        """
        ## turn white to black....
        annotations = np.where(annotations==255,0,annotations)
        annotations = np.expand_dims(annotations,-1)
        vectorized_annotation = np.array(np.equal(annotations,np.arange(self.num_classes)),dtype="float32")
        return vectorized_annotation
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        images = np.zeros([0]+self.image_size+[3],"float32")
        for image_filename in self.dataset[idx][0]:
            image = Image.open(image_filename).resize([self.image_size[1],self.image_size[0]])
            image = np.expand_dims(np.array(image,"float32"),0)
            if len(image.shape) == 3:
                image = np.expand_dims(image,-1)
                image = np.broadcast_to(image,[1]+self.image_size+[3])

            images = np.append(images,image,0)

        annotations = np.zeros([0]+self.image_size,"float32")
        for annotation_filename in self.dataset[idx][1]:
            annotation = Image.open(annotation_filename).resize([self.image_size[1],self.image_size[0]])
            annotation = np.array(annotation,"float32")
            annotation = np.expand_dims(annotation,0)
            annotations = np.append(annotations,annotation,0)

        annotations = self.annotation_processing(annotations)
        images/=255
        return images,annotations