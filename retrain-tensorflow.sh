#!/bin/bash

#Below is a script to retrain tensorflow using inceptionV3 pre trained model.
#tensor flow should be preinstalled.
#Training of new images takes place in last layer/fully connected layer.

echo -e "\n\nStarting retrain of tensorflow inception model"
current_dir=`dirname "$0"`
root_dir=`cd "${current_dir}";pwd`

#path to retrain files of tensorflow
retrain_dir="/home/ubuntu/udacity/tensorflow/tensorflow/examples/image_retraining"

python $retrain_dir/retrain.py \
--bottleneck_dir=$root_dir/bottlenecks \
--how_many_training_steps 1000 \
--model_dir=$root_dir/inception \
--output_graph=$root_dir/tf_files/retrained_graph.pb \
--output_labels=$root_dir/tf_files/retrained_labels.txt \
--image_dir $root_dir/data/train

echo -e "\n\nTraining ended"