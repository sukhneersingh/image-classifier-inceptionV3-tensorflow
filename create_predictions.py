import tensorflow as tf, sys
import os
import csv


#path to folder where test images are present
images_folder="/home/ubuntu/udacity/image-classifier-inceptionV3-tensorflow/data/test1/"

#path to submission file of kaggle
submission_file="/home/ubuntu/udacity/image-classifier-inceptionV3-tensorflow/data/submission.csv"

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#opening submisison_file and writing header
if submission_file:
        submission_file = open(submission_file, 'w', 1)
        w = csv.writer(submission_file,delimiter=",")
        w.writerow(["id","label"])
        
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    count=0
    for image_file in os.listdir(images_folder):
        count=count+1
        
        image_path = os.path.join(images_folder,image_file)
        
        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        
        prediction_value=0.5 
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
       
        if(predictions[0][0]>predictions[0][1]):
           prediction_value=1 #value for prediction as dog
        else:
           prediction_value=0 

        if submission_file:
            w.writerow([os.path.splitext(image_file)[0], str(predictions[0][0])])
            print("number of rows written : "+str(count))
        

if submission_file:
    print("Submission file created..")
    submisionfile.close()
           
        