# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import PIL
import urllib.request
from timm import create_model
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import torch 

def get_transforms(test_size=224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    Rs_size=int((256 / 224) * test_size)
    transformations = {}
    transformations['val_test'] = transforms.Compose(
        [transforms.Resize(Rs_size, interpolation=3),
         transforms.CenterCrop(test_size),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    return transformations


test_sizes=[224,224,224]
model_names =['deit_base_patch16_224-b5f2ef4d.pth',
              'deit_small_patch16_224-cd65a155.pth',
              'deit_tiny_patch16_224-a1311bcf.pth'
              ]
batch_sizes=[1045,2000,3800]
architecture_names=['deit_base_patch16_224','deit_small_patch16_224','deit_tiny_patch16_224']
paper_model_names = ['DeiT-base','DeiT-small','DeiT-tiny']
for test_size,mean_type,model_name,batch_size,architecture_name,paper_model_name in zip(test_sizes,mean_types,model_names,batch_sizes,architecture_names,paper_model_names):
    input_transform = get_transforms(test_size=test_size)['val_test']
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/deit/'+model_name, model_name)
    pretrained_dict=torch.load(str(model_name),map_location='cpu')['model']
    model = create_model(architecture_name, pretrained=False)
    model.load_state_dict(pretrained_dict)
    model.eval()
    
    # Run the benchmark
    with torch.no_grad():
        ImageNet.benchmark(
            model=model,
            model_description='DeiT',
            paper_model_name= paper_model_name,
            paper_arxiv_id='2012.12877',
            input_transform=input_transform,
            batch_size=batch_size,
            num_gpu=1
        )
    torch.cuda.empty_cache()
