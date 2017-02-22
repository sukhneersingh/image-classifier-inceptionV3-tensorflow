# Tensorflow animals classifier using inceptionV3 model

This repo is built for the challenge in video https://www.youtube.com/watch?v=cAICT4Al5Ow by siraj.

The use of this is to identify if the given image has Cat or Dog in it. 

Accuracy of the present model is 99%

##Requirements
- Tensorflow

##Usage
- Change path(retrain_dir) to tensorflow retrain directory in script "retrain-tensorflow.sh"
- Run the script retrain-tensorflow.sh
- Run the label_image script to label the image. python /tf_files/label_image.py <path_to_file>


This script uses pre built model inceptionV3 and retrain tensor flow with new categories of images for prediciton.
New data is present in data/train folder. Any number of new objects/categories can be trained by this script by just placing objects in this folder.
