# TextFreeKD-TIM
This repo is the official implementation of "**Text-Driven Medical Image Segmentation With Text-Free Inference via Knowledge Distillation**" [Paper Link](https://ieeexplore.ieee.org/document/10902555)

## Framework Overview
![Framework](IMG/Framework.png)

## Requirements
### 1. Environment
The implementation has been tested under the following environment:
```bash
python==3.8  
torch==1.12.1  
torchvision==0.13.1  
pytorch_lightning==1.9.0  
torchmetrics==0.10.3  
transformers==4.24.0  
monai==1.0.1  
pandas  
einops
```
### 2. Pretrained model

CXR-BERT-specialized see: https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized/tree/main

ConvNeXt-tiny see: https://huggingface.co/facebook/convnext-tiny-224/tree/main

Please download the required pretrained weights and place them in the directories 
specified by `bert_type` and `vision_type` in `config/training.yaml`:

```
MODEL:
  bert_type: ./lib/BiomedVLP-CXR-BERT-specialized
  vision_type: ./lib/convnext-tiny-224
```
## Dataset
1. QaTa-COV19 Dataset(images & segmentation mask)
QaTa-COV19 Dataset See Kaggle: https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset.
**We use QaTa-COV19-v2 in our experiments.**

2. QaTa-COV19 Text Annotations(from thrid party)
Check out the related content in LViT: https://github.com/HUANGLIZI/LViT

Thanks to Li et al. for their contributions. If you use this dataset, please cite their work.


