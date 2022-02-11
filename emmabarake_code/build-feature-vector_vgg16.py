#Emma Barake
#2021 Ecotone Waste CLassification Project

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import vgg16
import torchvision.transforms as transforms

# In[2]:


vgg16()


# In[2]:


image_paths = glob.glob('waste-classification-data/DATASET/*.jpg')
print(len(image_paths))


# In[3]:


print(image_paths[0])
img = Image.open(image_paths[0])
print(img.size)
img


# In[4]:


y = [os.path.split(image_path)[1].split('_')[0] for image_path in image_paths]
print(len(y))
print(y[:10])


# In[5]:


df = pd.DataFrame(data=list(zip(image_paths, y)), columns=['image_path', 'label'])


# In[6]:


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


# In[7]:


# Create python iterable list of images

image_paths = df['image_path']

# Transform from image to the input of CNN, following the same procedure as ImageNet

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image to 224x224
    transforms.ToTensor(), # convert pixel values to the range of [0,1]
    # normalize the pixel values according to the mean and std of ImageNet
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))]) 


# In[16]:


def load_image(path):
    '''Load a image and convert it to a pytorch Tensor as input to CNN'''
    img = Image.open(path) # Load the image with Pillow library
    img = img.convert('RGB') # Convert the image into RGB mode
    img = transform(img) # Transform the image
    return img

def get_feature(path):
    '''Run a pytorch Tensor through VGG16 and get feature vector '''
    img = load_image(path)
    img = img.unsqueeze(0) # make the tensor into single batch tensor with shape [1, 3, 224, 224]
    feature = model(img) # get feature
    feature = feature.detach().numpy() # detach the gradient, convert to numpy array
    return feature.flatten()


# In[ ]:


features = [] # create a list to store all the features
for image_path in tqdm(image_paths):
    feature = get_feature(image_path)
    features.append(feature)

print(np.array(features).shape)


# In[12]:


df['vgg16_fc1_feature'] = features
df.to_pickle('waste_features.pkl')


# In[17]:


df.head()


# In[ ]:




