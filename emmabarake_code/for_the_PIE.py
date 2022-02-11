#Emma Barake
#2021 Ecotone Waste CLassification Project

import os
import numpy as np
import glob
import pickle
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import vgg16
import torchvision.transforms as transforms

def import_classifier(filename):
    #get file location
    try:
        PATH = os.path.join(str(filename))
    except:
        return 'it didnt work...'
    #import classifier  
    unpickler = open(PATH, 'rb')
    classifier = pickle.load(unpickler)
    return classifier


clf = import_classifier('trained_SVC')
pca = import_classifier('fitted_pca_50')


# ## Set up CNN and PCA transformation function 
# **will be applied to imported image (see below)

# Define the model

class VGG_fc1(nn.Module):
    def __init__(self):
        super(VGG_fc1, self).__init__()
        self.features = vgg16(pretrained=True).features # convolutional layers
        self.avgpool = vgg16(pretrained=True).avgpool
        self.fc1 = vgg16(pretrained=True).classifier[0] # first layer of classifier
        
    def forward(self, x):
        """Extract first fully connected feature vector"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
    def classify(self, x):
        """Perform vgg16 inate classfication"""
        return self.classifier(x)
    
model = VGG_fc1().eval() # turn model into evaluation mode

# Transform from image to the input of CNN, following the same procedure as ImageNet

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image to 224x224
    transforms.ToTensor(), # convert pixel values to the range of [0,1]
    # normalize the pixel values according to the mean and std of ImageNet
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))])

def croppy_crop(image, pieces=4, transform_=True):
    '''take image an crop into specified number of pieces, number of pieces must always be a perfect square'''
    width, height = image.size
    target_height, target_width = height//int(np.sqrt(pieces)), width//int(np.sqrt(pieces))
    image_crop = [image.crop((y, x, y+target_width, x+target_height)) 
                  for x in range(0, height-1, target_height) 
                  for y in range(0, width-1, target_width)]
    if transform_:
        images_transformed = [transform(img) for img in image_crop]
        return images_transformed
    return image_crop

def load_image(path, transform=True):
    '''Load a image and convert it to a pytorch Tensor as input to CNN'''
    img = Image.open(path) # Load the image with Pillow library
    img = img.convert('RGB') # Convert the image into RGB mode
    img_list = croppy_crop(img, transform_=transform)
    return img_list

def get_feature(path):
    '''Run a pytorch Tensor through VGG16 and get feature vector '''
    features = []
    img_list = load_image(path)
    for img in img_list:
        img = img.unsqueeze(0) # make the tensor into single batch tensor with shape [1, 3, 224, 224]
        feature = model(img) # get feature
        feature = feature.detach().numpy() # detach the gradient, convert to numpy array
        features += [feature.flatten()]
    return features


# ## Import Image to classify and convert to feature vector
# will need to change when implementing into the rassspppppberry pi...
# image_paths = glob.glob('test photos/test*.jpg')
image_paths = glob.glob('test_photos_1/test*.jpg')
print(len(image_paths))


def run_classification(image_path_list, clf, pca):
    features = []
    images = []
    for path in image_path_list:
        features+= get_feature(path)
        images+= load_image(path, transform=False)
    features_pca = pca.transform(features)
    predicted = clf.predict(features_pca)
    return predicted, images

predicted, images = run_classification(image_paths, clf, pca)

print(predicted)

# got 11/12 correct! The image that was not classified correctly had a white plastic bag which is the same color as the background, so it likely ignored it thinking it was part of the background and only classified the rest of the image
