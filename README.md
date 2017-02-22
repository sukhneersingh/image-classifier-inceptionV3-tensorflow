# Tensorflow animal's classifier using inceptionV3 model

This repo is built for the challenge in video https://www.youtube.com/watch?v=cAICT4Al5Ow by siraj.

The use of this is to identify if the given image has Cat or Dog in it. 

Accuracy of the present model is 99% because of inception model used.

![Alt text](/screenshots/accuracy.PNG?raw=true "Optional Title")


This script uses pre built model inceptionV3 and retrain tensor flow with new categories of images for prediciton.
New data is present in data/train folder. Any number of new objects/categories can be trained by this script by just placing objects in this(train) folder.


##Requirements
- Tensorflow
- python

##Usage
- Change path(retrain_dir) to tensorflow retrain directory in script "retrain-tensorflow.sh"
- Run the script "retrain-tensorflow.sh"
- Run the label_image.py python script with argument as path to image. 
