# Partial Convolutions for Image Inpainting using Keras
Keras implementation of "*Image Inpainting for Irregular Holes Using Partial Convolutions*", https://arxiv.org/abs/1804.07723. A huge shoutout the authors Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao and Bryan Catanzaro from NVIDIA corporation for releasing this awesome paper, it's been a great learning experience for me to implement the architecture, the partial convolutional layer, and the loss functions. 

<img src='./data/images/sample_results.png' />

# Dependencies
* Python 3.6
* Keras 2.2.4
* Tensorflow 1.12

# How to use this repository
The easiest way to try a few predictions with this algorithm is to go to [www.fixmyphoto.ai](https://www.fixmyphoto.ai/), where I've deployed it on a serverless React application with AWS lambda functions handling inference.

If you want to dig into the code, the primary implementations of the new `PConv2D` keras layer as well as the `UNet`-like architecture using these partial convolutional layers can be found in `libs/pconv_layer.py` and `libs/pconv_model.py`, respectively - this is where the bulk of the implementation can be found. Beyond this I've set up four jupyter notebooks, which details the several steps I went through while implementing the network, namely:

Step 1: Creating random irregular masks<br />
Step 2: Implementing and testing the implementation of the `PConv2D` layer<br />
Step 3: Implementing and testing the UNet architecture with `PConv2D` layers<br />
Step 4: Training & testing the final architecture on ImageNet<br />
Step 5: Simplistic attempt at predicting arbitrary image sizes through image chunking

## Pre-trained weights
I've ported the VGG16 weights from PyTorch to keras; this means the `1/255.` pixel scaling can be used for the VGG16 network similarly to PyTorch. 
* [Ported VGG 16 weights](https://drive.google.com/open?id=1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0)
* [PConv on Imagenet](https://drive.google.com/open?id=1OdbuNJj4gV9KUoknQG063jJrJ1srhBvU)
* PConv on Places2 [needs training]
* PConv on CelebaHQ [needs training]

If you have Image (with holes) and Mask available with you then Mytest.py can be used to test the model with pre-trained weights. To get the pretrained weights, follow this link https://drive.google.com/file/d/1OdbuNJj4gV9KUoknQG063jJrJ1srhBvU/view and paste it in the PConv directory only.
Now, Follow the instructions and code given in Myfile.py and have fun.



## Training on your own dataset
You can either go directly to [step 4 notebook](https://github.com/MathiasGruber/PConv-Keras/blob/master/notebooks/Step4%20-%20Imagenet%20Training.ipynb), or alternatively use the CLI (make sure to download the converted VGG16 weights):
```
python main.py \
    --name MyDataset \
    --train TRAINING_PATH \
    --validation VALIDATION_PATH \
    --test TEST_PATH \
    --vgg_path './data/logs/pytorch_to_keras_vgg16.h5'
```

## Training Procedure
Network was trained on ImageNet with a batch size of 1, and each epoch was specified to be 10,000 batches long. Training was furthermore performed using the Adam optimizer in two stages since batch normalization presents an issue for the masked convolutions (since mean and variance is calculated for hole pixels).

**Stage 1**
Learning rate of 0.0001 for 50 epochs with batch normalization enabled in all layers

**Stage 2**
Learning rate of 0.00005 for 50 epochs where batch normalization in all encoding layers is disabled.

Training time for shown images was absolutely crazy long, but that is likely because of my poor personal setup. The few tests I've tried on a 1080Ti (with batch size of 4) indicates that training time could be around 10 days, as specified in the paper.
