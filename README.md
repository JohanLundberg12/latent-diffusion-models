# About

Implementation of the [DDPM](https://papers.labml.ai/paper/2006.11239) and [https://papers.labml.ai/paper/1512.03385](ResNet) for data augmentation to
improve an image classifier. 

# Usage

To train the diffusion model run ``python train_diffusion_model.py config_files/config_file.yaml´´

To generate more data using the trained diffusion model run ``python generate_images.py config_files/config_file.yaml´´

To train the resnet classifier run ``python train_resnet_classifier.py config_files/config_file.yaml´´
