# About

Implementation of the [DDPM](https://papers.labml.ai/paper/2006.11239) and [ResNet](https://papers.labml.ai/paper/1512.03385) for data augmentation to
improve an image classifier. 

# Usage

To train the diffusion model run 
```bash
$ python train_diffusion_model.py config_files/config_file.yaml
```

To generate more data using the trained diffusion model run 
```bash
$ python generate_images.py config_files/config_file.yaml
```

To train the resnet classifier run 
```bash
$ python train_resnet_classifier.py config_files/config_file.yaml
```
