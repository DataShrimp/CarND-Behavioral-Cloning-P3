
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Model Visualization"
[image2]: ./examples/image2.jpg "Center Driving"
[image3]: ./examples/image3.jpg "Recovery Image"
[image4]: ./examples/image4.jpg "Recovery Image"
[image5]: ./examples/image5.jpg "Recovery Image"
[image6]: ./examples/image6.jpg "Normal Image"
[image7]: ./examples/image7.jpg "Flipped Image"
[image8]: ./examples/image8.png "Model Mean Squared Error Loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 128 (model.py lines 90-100) 

The model includes RELU layers to introduce nonlinearity (code line 90-94), and the data is normalized in the model using a Keras lambda layer (code line 84). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 99). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 105-109). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 103).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And I totally collected three laps of lane driving. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a proven model and then fine-tune it to successfully autonomous driving in simulated environment. 

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it was used for training a real car to drive autonomously. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added a dropout layer with probability of 0.8 to the model so that the mean squared error on the validation set would be low as well. 

Then I tried several hyper-parameters to observe which could achieve a better result according to the mean squared error on both training and validation sets. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In order to improve the driving behavior in these cases, I collected more data from one laps to three laps, which contains one lap of recovery driving from the sides and one lap of smoothly driving. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-100) consisted of a convolution neural network with the following layers and layer sizes.
1. 5x5 Convolution layer, 24 kernels, Relu activation
2. 5x5 Convolution layer, 36 kernels, Relu activation
3. 5x5 Convolution layer, 48 kernels, Relu activation
4. 3x3 Convolution layer, 64 kernels, Relu activation
5. 3x3 Convolution layer, 128 kernels, Relu activation
6. Flattern layer
7. Fully Connected layer, outputs 128
8. Fully Connected layer, outputs 64
9. Fully Connected layer, outputs 32
10. Dropout layer, probability of 0.8
11. Fully Connected layer, outputs 1

Here is a visualization of the architecture. 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when driving to the right or left sides. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help to take the opposite sign of steering measurement. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also added left and right camera images to the training set with a correction parameter of 0.2 to the steering angle. 

After the collection process, I had 5680x2x3=34080 number of data points. I then preprocessed this data by normalized the pixel value to (0,1) and cropped each image to focus on only the portion of image that is useful for predicting a steering angle. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by both the training loss and the validation loss are low to 0.25 and 0.15. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The mean squared error loss of the model is shown as followed. 

![alt text][image8]
