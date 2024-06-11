import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset_utils import crop_and_resize, combine_and_mask
import matplotlib.pyplot as plt
import resnet_model
import torch
import torch.nn as nn
from torchvision import models


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft

    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)

            x = self.model.bn1(x)

            x = self.model.relu(x)

            x = self.model.maxpool(x)

            x = self.model.layer1(x)

            x = self.model.layer2(x)

            x = self.model.layer3(x)

            x = self.model.layer4(x)

            x = self.model.avgpool(x)

            x = x.view(x.size(0), -1)
            x = self.model.fc(x)

        return x


################ Paths and other configs - Set these #################################
def generate(r_water,r_land,n_sample,cub_dir = './CUB',places_dir = './data_large'):
    #r_water: 水鸟在水环境的概率
    #r_land: 陆鸟在水环境的概率
    
    

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds
    ######################################################################################

    images_path = os.path.join(cub_dir, 'images.txt')

    df = pd.read_csv(
        images_path,
        sep=" ",
        header=None,
        names=['img_id', 'img_filename'],
        index_col='img_id')

    ### Set up labels of waterbirds vs. landbirds
    # We consider water birds = seabirds and waterfowl.
    water_birds_list = [
        'Albatross', # Seabirds
        'Auklet',
        'Cormorant',
        'Frigatebird',
        'Fulmar',
        'Gull',
        'Jaeger',
        'Kittiwake',
        'Pelican',
        'Puffin',
        'Tern',
        'Gadwall', # Waterfowl
        'Grebe',
        'Mallard',
        'Merganser',
        'Guillemot',
        'Pacific_Loon'
    ]

    wb=[]
    lb=[]
    for species_name in df['img_filename']:
        iswater=0
        species_name_new=species_name.split('/')[0].split('.')[1].lower()
        for water_bird in water_birds_list:
            if water_bird.lower() in species_name_new:
                wb.append(species_name)
                iswater=1
        if iswater==0:
            lb.append(species_name)

    resnet50=ft_net()
    x=torch.zeros(n_sample,1000)
    y=torch.zeros(n_sample)
    z=torch.zeros(n_sample)
    for i in range(n_sample):
        # Load bird image and segmentation
        iswater = np.random.binomial(1,0.5,1)
        if iswater[0]==1:
            bird_name=random.choice(wb)
        else:
            bird_name=random.choice(lb)
        img_path = os.path.join(cub_dir, 'images', bird_name)
        seg_path = os.path.join(cub_dir, 'segmentations', bird_name.replace('.jpg','.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        # Skip front /
        #print(df.loc[i])
        if iswater[0]==1:
            background= np.random.binomial(1,r_water,1)
        else:
            background= np.random.binomial(1,r_land,1)
        background_path=random.choice(target_places[background[0]])
        num_background=str(random.randint(0,5000)).rjust(8,'0')+".jpg"

        place_path = os.path.join(places_dir, background_path[0],background_path,num_background)
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = combine_and_mask(place, seg_np, img_black)
        combined_img = torch.transpose(torch.tensor(combined_img,dtype=torch.float32),0,2)
        combined_img= torch.unsqueeze(combined_img, dim=0)
        #combined_img = torch.reshape(combined_img,(3,-1))
        #print(combined_img.shape)
        feature=resnet50(combined_img)
        y[i]=iswater[0]  #0: land 1: water
        z[i]=background[0] #0: land 1:water
        x[i]=feature
    return x,y,z


x0,y0,z0=generate(0.5,0.5,5)
print(x0,y0)
