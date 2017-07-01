#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

[//]: # (Image References)

[image0]: ./report_images/dataset.png "Dataset"
[image1]: ./report_images/grayscaled.png "Grayscaling"
[image2]: ./report_images/normalized.png "Normalizing"
[image3]: ./report_images/internet.png "Normalizing"

---
###Writeup / README

This README file will explain how to understand the project.
On the GitHub repositery you're in, you can see multiple directories and files: 
* Traffic_Signs_classifier.html or .ipynb are my code files in two formats. You can use either one of them.
* signnames.csv contain a CSV file with each traffic sign code (for example 1) and its signification (for example "20km/h limitation). The dataset contains 43 possible outputs.

That's it ! I did not include the dataset because it is way too heavy and the correctors of this project have it already. Just know that you can download it on : https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

Concerning the README, You're reading it! and here is a link to my project (https://github.com/Jeremy26/traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. DATA SET SUMMARY
You can see it more clearly on the code, but the dataset (51839 images) includes three subsets :
* Training files, which correspond approximately to 67% of the whole dataset
* Validation files, about 9% of the dataset
* Test files, about 24% of the dataset

Validation is used to verify the model and will be used to get the accuracy up. Testing is never seen data, used to validate validation.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

![alt text][image0]

Obviously, training set is blue, testing set is green, validation set is red

###Design and Test a Model Architecture

####1. PREPROCESSING

When training a model, there is an important part called preprocessing. I used two proprocessing techniques :
* Grayscale
* Normalization

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1]


As a last step, I normalized the image data because I wanted to have values in the same range. This help having good weights and biases/
After normalizing, the image (grayscaled first) is :
![alt text][image2]

I did not augment my training set because I didn't think it was needed. Looking at the final solution (with the internet files), I think that could have helped getting better percentage.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of 5 layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image   							| 
| LAYER 1         		|   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|			Activation function									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| LAYER 2         		|    							| 
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x6			|
| RELU					|				Activation function								|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| 5x5x16 gives an output of 400 				|
| LAYER 3        		|    							| 
| Fully connected		| output 120        									|
| RELU				|        									|
|	DROPOUT				|	keep_prob 1.0 for validation, 0.5 for evaluation				|
| LAYER 4						|												|
|	Fully Connected			|		output 84										|
|	RELU					|		Activation function										|
|	DROPOUT					|		keep_prob 1.0 for validation, 0.5 for evaluation		|
| LAYER 5						|												|
|	Fully Connected					|			output 43								|

We now have our 43 possible outputs from a 32x32x1 image !

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

You could notice I used two times dropout function in my architecture. It really made my network more robust and increased the probability from 1 to 2% each time I used it. I didn't want to use it more not to destroy too much of the network.
The parameters taken were :
* A low learning rate : 0.001 that could be lower.
* 30 Epochs (enough to train the model up to 96%)
* mu = 0, unchanged from the Lab
* sigma = 0,13 : I was surprised how much changing the sigma increased the accuracy
* BATCH SIZE : 64 - I tried really high and really low, 64 seems to be the best.

I finally trained the model and saved it.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 96.8%
* test set accuracy of 95.0%

In the end, the LeNet architecture is working well. The dropout made it robust to change.

###Test a Model on New Images

####1. 10 INTERNET SIGNS

Here are five German traffic signs that I found on the web:

![alt text][image3]

These images should be okay to classify. Maybe the 70km/h will give difficulties due to the zoom out.

####2. PREDICTIONS ANALYSIS

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians      		| Turn Left Ahead   									| 
| Turn Right Ahead     			| Turn Right Ahead 										|
| Yield					| Yield											|
| STOP	      		| Priority Road					 				|
| Road work			| Road work      							|
| Speed limit (30km/h)			| Slippery road     							|
| No entry			| Speed limit (100km/h)      							|
| Speed limit (70km/h)			| Speed limit (30km/h)      							|
| Go straight or right			| Turn right ahead     							|
| Priority road			| Priority road      							|

The model was able to correctly guess 4 of the 10 traffic signs, which gives an accuracy of 40%. 
This compares not favorably to the accuracy on the test set of 94%.
I think the problem is in the data augmentation, needed.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability			        |     Prediction	        					                   | Result |
|:---------------------:|:---------------------------------------------:| :-------:|
| 0.5804      		| Turn Left Ahead   									| Wrong
| 1.0000    			| Turn Right Ahead 										| Right
| 1.0000					| Yield											| Right
| 1.0000	      		| Priority Road					 				| Wrong
| 1.0000			| Road work      							| Right
| 0.2209			| Slippery road     							| Wrong
| 0.4874			| Speed limit (100km/h)      							| Wrong
| 0.4015			| Speed limit (30km/h)      							| Wrong
| 0.9997			| Turn right ahead     							| Wrong
| 1.0000		| Priority road      							| Right


When the prediction is 99% or 100%, it is always right. Except for the Priority Road that was wrong with 100% chance to be right. That makes me wonder why...
Generally, when it is wrong, it gives very low percentage to the actual sign.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Would love to see a solution for this.
