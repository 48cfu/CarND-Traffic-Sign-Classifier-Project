# **Traffic Sign Recognition** 

Here is a link to my [project code](./source/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/histogram.png "Visualization"
[image2]: ./data/example_input.png "Input example"
[intimage1]: ./data/intermediate_layers.png "Intermediate layer 1"
[intimage2]: ./data/intermediate_layers_1.png "Intermediate layer 2"
[image4]: ./images/newsign1.jpg "Traffic Sign 1"
[image5]: ./images/newsign2.jpg "Traffic Sign 2"
[image6]: ./images/newsign3.jpg "Traffic Sign 3"
[image7]: ./images/newsign4.jpg "Traffic Sign 4"
[image8]: ./images/newsign5.jpg "Traffic Sign 5"
[image9]: ./images/newsign6.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is not uniformly distributed. Some classes clearly have more training and validation input images.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to scale the image to (image - 128)/128, because we want to avoid numerical issues and also have a zero-mean input with fixed variance. The initial 3 channels were also fused and made just one. As deciding in the RGB domain it's more prone to failure in this domain.

I did not augment the initial dataset because a rotation of for example 90 degrees may create a wrong answer ('turn left' and 'turn right' may be confused with 'straight only' once rotated). I was not sure of other technique for augmenting the initial dataset.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (borrowed from the LeNet architecture, and added regularization to avoid overfitting):

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU          		|              									|
| DROPOUT   			| 		                            			|
| FULLY CONNECT LAYER	| input 5x5x16, output 120						|
| RELU					|												|
| DROPOUT   			| 		                            			|
| FULLY CONNECT LAYER	| input 120, output 84			    			|
| RELU					|												|
| DROPOUT   			| 		                            			|
| FULLY CONNECT LAYER	| input 84, output 43 = n_classes    			|
| RELU					|												|
| LOGITS      			| 		                            			|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Hyperparameter   		|     Choice    	        					| 
|:---------------------:|:---------------------------------------------:| 
| Epochs        		| 25   							                | 
| Batch size         	| 128                                          	|
| Learning rate 		| 0.001											|
| Keep probability     	| 0.7                           				|

Naturally, the keep probability of the dropout regularizer used during evaluation is 1.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.964
* test set accuracy of 0.949

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose SIX German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The second image might be difficult to classify because there is another traffic sign behind it, which might make the deep CNN think the traffic sign is round. The first layers in the architecture may be activated by the curvy lines behind it. Similar reasoning for the last picture since a piece of another traffic sign is visible, and the position in the image doesn't matter (yayy CNN). Analogously, the fourth traffic sign is not isolated, and therefore the lower layers of the deep CNN will defininitel be activated due to the ambulance.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| No passing  									| 
| Priority road    		| Priority road 								|
| Stop				    | Stop											|
| Speed limit (30km/h)	| Speed limit (30km/h)			 				|
| Road work			    | Road work     					    		|


The model was able to correctly guess 6 out of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 99.6%. However the pictures used were in good condition. It may be interesting to observe what happens with traffic sign in very bad condition but still recognizable by humans. The current architecture must be revisited.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is **very** sure that this is a **No passing** sign (probability of 0.999898), and the image does contain a **No passing** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999898      		| No passing  									| 
| 9.32169e-05       	| No passing for vehicles over 3.5 metric tons	|
| 9.10213e-06		    | End of no passing								|
| 5.24268e-08	      	| No entry					 				    |
| 3.19526e-09			| Dangerous curve      							|


For the second image, , the model is relatively sure that this is a **Priority road** sign (probability of 0.999972), and the image does contain a **Priority road** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999972      		| Priority road 								| 
| 2.3872e-05          	| Roundabout mandatory	                        |
| 1.24534e-06		    | Yield 								        |
| 7.90884e-07	      	| No passing					 				|
| 7.72734e-07			| End of all speed and passing limits      		|

The remaining traffic signs follow a similar distribution of the top 5 highest probabilities. The details of all the others are in the [Jupyter notebook](./source/Traffic_Sign_Classifier.ipynb) ./source/Traffic_Sign_Classifier.ipynb


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Below is an example of an input from the dataset(averaged RGB, but not normalized)

![alt text][image2] <br />

The output of the first layer 
Layer 1 <br />
![alt text][intimage1] <br />
The output of the secondo layer
Layer 2<br />
![alt text][intimage2]

I'm not sure how to interpret the outputs. Looks like the first layer is already being activated by the triangulare shape, and not lines and so one. This might come from the fact that the LeNet is not so deep?
