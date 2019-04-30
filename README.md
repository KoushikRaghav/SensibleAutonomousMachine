# Sensible Autonomous Machine #

## INTRODUCTION ##
In recent times, automation has achieved improvements by quality, accuracy and precision. At the same time, the technology itself continues to evolve, bringing new waves of advances in robotics, analytics, and artificial intelligence (AI), and especially machine learning. Together they amount to a step change in technical capabilities that could have profound implications for business, for the economy, and more broadly, for society.  

The modules used in this program include

### 1.1	 Robotics ### 
Robotics is a branch of engineering which incorporates multiple disciplines to design, build, program and use robotic machines.  Robots are used in industries for speeding up the manufacturing process. AI is a highly useful tool in robotic assembly applications. When combined with advanced vision systems, AI can help with real-time course correction, which is particularly useful in complex manufacturing sectors. A handful of robotic systems are now being sold as open source systems with AI capability. Users train robots to do custom tasks based on their specific application, such as small-scale agriculture. The convergence of open source robotics and AI could be a huge trend in the future of AI robots.

### 1.2 Automation in Machines ###

Automation is any individual involved in the creation and application of technology to monitor and control the production and delivery of products and services. Automakers are moving at a frenzied pace to add more and more intelligence to vehicles developed a scale to describe the six different levels of automation for self-driving cars. Level 0 is the driver actually steps on the gas to go faster, steps on the brake to slow down and uses the steering wheel to turn. Level 1 is the driver is still in control of the overall operation and safety of the vehicle. Level 2 is the driver is still responsible for the safe operation of the vehicle. Level 3 states that the car can drive itself, but the human driver must still pay attention and be prepared to take over at any time. Level 4 explains that the car can be driven by a person, but it doesn’t always need to be. It can drive itself full-time under the right circumstances. Level 5 proposes that the car controls itself under all circumstances with no expectation of human intervention. 

### 1.3 Deep Learning ###

Deep Learning is a new area of Machine Learning research, which has been introduced with the objective of moving Machine Learning closer to one of its original goals, Artificial Intelligence. Deep Learning provides computers with the ability to learn without being explicitly programmed. Deep learning focuses on the development of computer programs that can change when exposed to new data.  The process of Deep Learning is similar to that of data mining. Both systems search through data to look for patterns.

### 1.4	 RCNN ###
 
Region-CNN (R-CNN) is one of the state-of-the-art CNN-based deep learning object detection approaches. Based on this, there are fast R-CNN and faster RCNN for faster speed object detection as well as mask R-CNN for object instance segmentation. On the other hand, there are also other object detection approaches, such as YOLO and SSD. 

### 1.5 Python-3.5.2 ###

Python’s standard library is very extensive, offering a wide range of facilities as indicated by the long table of contents listed below. The library contains built-in modules (written in C) that provide access to system functionality such as file I/O that would otherwise be inaccessible to Python programmers, as well as modules written in Python that provide standardized solutions for many problems that occur in everyday programming. Some of these modules are explicitly designed to encourage and enhance the portability of Python programs by abstracting away platform-specifics into platform-neutral APIs.

### 1.6	 TensorFlow 1.12.0 ###

TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.

### 1.7	 Pip ###

pip is a package management system used to install and manage software packages written in Python. Many packages can be found in the default source for packages and their dependencies —Python Package Index (PyPI). 
Python 2.7.9 and later (on the python2 series), and Python 3.4 and later include pip (pip3 for Python 3) by default. pip is a recursive acronym that can stand for either "Pip Installs Packages" or "Pip Installs Python". Alternatively, pip stands for "preferred installer program".

### 1.8	 NumPy ###

NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. The ancestor of NumPy, Numeric, was originally created by Jim Hugunin with contributions from several other developers. In 2005, Travis Oliphant created NumPy by incorporating features of the competing num array into Numeric, with extensive modifications. NumPy is open-source software and has many contributors.

## OVERVIEW ##

The main idea of the project is to initiate a self-driving machine, observing the surroundings across a transport region and act accordingly by the provider’s instructions. This initiative brings many real-world things to autonomous creature and the main purpose is to save time from user’s point of view. 

### 2.1 Primary goal ###

Interfacing SAM to follow back an object by crossing obstacles and to reach destination by sensing the response signal from Smartphone. Monitoring entire surrounding to identify the color patterns of traffic signal and performing right/left turn by getting trained in predicting static/dynamic models. Maintain ratio of frames captured and running status of machine.
 
### 2.2 Secondary Goal ###

![2](https://user-images.githubusercontent.com/24918359/56951544-bedc1f80-6b55-11e9-9130-c8da5accd3aa.JPG) 

## ARCHITECTURE DIAGRAM ##

![3](https://user-images.githubusercontent.com/24918359/56951976-c5b76200-6b56-11e9-84fb-0974f7e516e1.jpg)

## MODULE DESCRIPTION ##

1) Processing Real Time Data
2)	Calibration of Machine by embedding sensors
3)	Manipulation of the Machine
4) Integration of the Machine with Mobile Application

### 3.1 Processing Real Time Data ###

*	Pi camera is integrated with Raspberry Pi.
*	Image processing is done in real time scenario by recognizing object patterns and detecting traffic light using R-CNN.
*	Deep Learning implements feed-forward artificial neural networks or, more particularly, multi-layer perceptron (MLP), the most commonly used type of neural networks. MLP consists of the input layer, output layer, and one or more hidden layers. 
*	Each layer of MLP includes one or more neurons directionally linked with the neurons from the previous and the next layer.

![4](https://user-images.githubusercontent.com/24918359/56951984-cd770680-6b56-11e9-9087-ae2db5096b56.jpg)

### 3.2 Calibration of the Machine by embedding sensors###

### 3.2.1 Detection of Obstacles	###

* Absorption and Reflection of black and white signals to make the machine sense the road patterns for moving forward and turn operations (left and right).

![5](https://user-images.githubusercontent.com/24918359/56951990-d1a32400-6b56-11e9-8447-28f125895c5d.jpg)

### 3.2.2	Measurement of distance between objects ###

* Measuring the distance between the machine and other objects on-road by sensing frequency through Ultrasonic Sensor.

![6](https://user-images.githubusercontent.com/24918359/56952001-d5cf4180-6b56-11e9-87b6-1876523f667b.jpg)

### 3.3	Manipulation of the Machine ###

* L293D Motor IC is interfaced with Raspberry PI to perform start and stop operations.

![7](https://user-images.githubusercontent.com/24918359/56952009-d8ca3200-6b56-11e9-9e6c-6797e0520cb4.jpg)

### 3.4	Integration of the Machine with Mobile Application ###

* User provides the Geo Location via Android Application to the Machine where Latitude and Longitude data is fed to the machine to reach destination.

![8](https://user-images.githubusercontent.com/24918359/56952013-db2c8c00-6b56-11e9-9878-e0b2318743a2.jpg)

## PROCESSING AND TRAINING ##

### Installing TensorFlow ###

### 1. Update the Raspberry Pi ###

First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:

sudo apt-get update
sudo apt-get dist-upgrade

![1](https://user-images.githubusercontent.com/24918359/56953172-8dfde980-6b59-11e9-90bd-ad1ec2777631.png)

### 2. Install TensorFlow ###

Next, we’ll install TensorFlow. In the /home/pi directory, create a folder called ‘tf’, which will be used to hold all the installation files for TensorFlow and Protobuf, and cd into it:

mkdir tf
cd tf
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.8.0/tensorflow-1.12.0-cp35-none-linux_armv7l.whl

sudo pip3 install /home/pi/tf/tensorflow-1.12.0-cp35-none-linux_armv7l.whl

TensorFlow also needs the LibAtlas package. Install it by issuing 

sudo apt-get install libatlas-base-dev

While we’re at it, let’s install other dependencies that will be used by the TensorFlow Object Detection API. These are listed on the installation instructions in TensorFlow’s Object Detection GitHub repository. Issue:

sudo pip3 install pillow lxml jupyter matplotlib cython
sudo apt-get install python-tk

### 3. Install OpenCV ###

TensorFlow’s object detection examples typically use matplotlib to display images, but I prefer to use OpenCV because it’s easier to work with and less error prone. The object detection scripts in this guide’s GitHub repository use OpenCV. So, we need to install OpenCV.

To get OpenCV working on the Raspberry Pi, there’s quite a few dependencies that need to be installed through apt-get. If any of the following commands don’t work, issue “sudo apt-get update” and then try again. Issue:

sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install qt4-dev-tools

Now that we’ve got all those installed, we can install OpenCV. Issue:

pip3 install opencv-python

### 4. Compile and Install Protobuf ###

The TensorFlow object detection API uses Protobuf, a package that implements Google’s Protocol Buffer data format. Unfortunately, there’s currently no easy way to install Protobuf on the Raspberry Pi. We have to compile it from source ourselves and then install it.

First, get the packages needed to compile Protobuf from source. Issue:

sudo apt-get install autoconf automake libtool curl

Then download the protobuf release from its GitHub repository by issuing:

wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz

If a more recent version of protobuf is available, download that instead. Unpack the file and cd into the folder:

tar -zxvf protobuf-all-3.5.1.tar.gz
cd protobuf-3.5.1

Configure the build by issuing the following command (it takes about 2 minutes):

./configure

Build the package by issuing:

make

When it’s finished, issue:

make check 

This process takes even longer, clocking in at 107 minutes on Pi. According to other guides I’ve seen, this command may exit out with errors, but Protobuf will still work. Now that it’s built, install it by issuing:

sudo make install

Then move into the python directory and export the library path: 
cd python
export LD_LIBRARY_PATH=../src/.libs

Next, issue:
python3 setup.py build --cpp_implementation 
python3 setup.py test --cpp_implementation
sudo python3 setup.py install --cpp_implementation

Then issue the following path commands:

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3

Finally, issue:

sudo ldconfig

Now Protobuf is installed on the Pi. Verify it’s installed correctly by issuing the command below and making sure it puts out the default help text.

Protoc

For some reason, the Raspberry Pi needs to be restarted after this process, or TensorFlow will not work. Go ahead and reboot the Pi by issuing:

sudo reboot now

### 5. Set up TensorFlow Directory Structure and PYTHONPATH Variable ###

Now that we’ve installed all the packages, we need to set up the TensorFlow directory. Move back to your home directory, then make a directory called “tensorflow1”, and cd into it.

mkdir tensorflow1
cd tensorflow1

Download the tensorflow repository from GitHub by issuing:

git clone --recurse-submodules https://github.com/tensorflow/models.git

Next, we need to modify the PYTHONPATH environment variable to point at some directories inside the TensorFlow repository we just downloaded. We want PYTHONPATH to be set every time we open a terminal, so we have to modify the .bashrc file. Open it by issuing:
sudo nano ~/.bashrc

export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow1/models/research:/home/pi/tensorflow1/models/research/slim

![2](https://user-images.githubusercontent.com/24918359/56953173-8dfde980-6b59-11e9-99d4-6fcecf96bf92.png)

Then, save and exit the file. This makes it so the “export PYTHONPATH” command is called every time you open a new terminal, so the PYTHONPATH variable will always be set appropriately. Close and then re-open the terminal.

Now, we need to use Protoc to compile the Protocol Buffer (.proto) files used by the Object Detection API. The .proto files are located in /research/object_detection/protos, but we need to execute the command from the /research directory. Issue:

cd /home/pi/tensorflow1/models/research
protoc object_detection/protos/*.proto --python_out=.

This command converts all the "name".proto files to "name_pb2".py files. Next, move into the object_detection directory:

cd /home/pi/tensorflow1/models/research/object_detection

Now, we’ll download the SSD_Mobilenet model from the TensorFlow detection model zoo. The model zoo is Google’s collection of pre-trained object detection models that have various levels of speed and accuracy. The Raspberry Pi has a weak processor, so we need to use a model that takes less processing power. Though the model will run faster, it comes at a tradeoff of having lower accuracy. For this tutorial, we’ll use SSD-MobileNet, which is the fastest model available.

Google is continuously releasing models with improved speed and performance, so check back at the model zoo often to see if there are any better models.

Download the SSD-MobileNet model and unpack it by issuing:

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

Now the model is in the object_detection directory and ready to be used.

### 6. Detect Objects ###

Okay, now everything is set up for performing object detection on the Pi! The Python script in this repository, Object_detection_picamera.py, detects objects in live feeds from a Picamera or USB webcam. Basically, the script sets paths to the model and label map, loads the model into memory, initializes the Picamera, and then begins performing object detection on each video frame from the Picamera.

If you’re using a Picamera, make sure it is enabled in the Raspberry Pi configuration menu.

![3](https://user-images.githubusercontent.com/24918359/56953174-8dfde980-6b59-11e9-9563-c660969fbe23.png)

Download the Object_detection_picamera.py file into the object_detection directory by issuing:

wget https://raw.githubusercontent.com/Parthi_Koushik/TensorFlow-Object-Detection-on-the-Raspberry-Pi/master/Object_detection_picamera.py

Run the script by issuing:

python3 Object_detection_picamera.py 

The script defaults to using an attached Picamera. If you have a USB webcam instead, add --usbcam to the end of the command:
python3 Object_detection_picamera.py –usbcam

Once the script initializes (which can take up to 30 seconds), you will see a window showing a live view from your camera. Common objects inside the view will be identified and have a rectangle drawn around them.

![4](https://user-images.githubusercontent.com/24918359/56953175-8e968000-6b59-11e9-812f-8067e6aac3c0.png)

You can also use a model by adding the frozen inference graph into the object_detection directory and changing the model path in the script. You can test this out using my playing card detector model (transferred from ssd_mobilenet_v2 model and trained on TensorFlow v1.12). Once downloaded and extracted the model, or if you have your own model, place the model folder into the object_detection directory. Place the label_map.pbtxt file into the object_detection/data directory.

![5](https://user-images.githubusercontent.com/24918359/56953176-8e968000-6b59-11e9-9fce-152c5938dc39.png)

Then, open the Object_detection_picamera.py script in a text editor. Go to the line where MODEL_NAME is set and change the string to match the name of the new model folder. Then, on the line where PATH_TO_LABELS is set, change the name of the labelmap file to match the new label map. Change the NUM_CLASSES variable to the number of classes your model can identify.   

![6](https://user-images.githubusercontent.com/24918359/56953177-8e968000-6b59-11e9-89c9-b7573f63d4ff.png)

Now, when you run the script, it will use your model rather than the SSDLite_MobileNet model. 

### Training Datasets ###

### 1. Set up TensorFlow Directory and Anaconda Virtual Environment ###

The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model.

### 1a. Download TensorFlow Object Detection API repository from GitHub ###

Create a folder directly in C: and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.

### 1b. Download the SSD_MobileNet-V2-COCO model from TensorFlow's model zoo ###

TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo. Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy. Use the SSD-MobileNet model for training.

### 1c. Download repository ###

Download the full repository located on this page (scroll to the top and click Clone or Download) and extract all the contents directly into the C:\tensorflow1\models\research\object_detection directory. (You can overwrite the existing "README.md" file.) This establishes a specific directory structure that will be used.

![7](https://user-images.githubusercontent.com/24918359/56953179-8e968000-6b59-11e9-9b27-2a64e62dcaf5.jpg)

This contains the images, annotation data, .csv files, and TFRecords needed to train a "Pinochle Deck" playing card detector. You can use these images and data to practice making your own Pinochle Card Detector. It also contains Python scripts that are used to generate the training data. It has scripts to test out the object detection classifier on images, videos, or a webcam feed. You can ignore the \doc folder and its files; they are just there to hold the images used for this readme.

If you want to practice training your own "Pinochle Deck" card detector, you can leave all the files as they are. You can follow along with this tutorial to see how each of the files were generated, and then run the training. You will still need to generate the TFRecord files (train.record and test.record) as described in Step 4.

If you want to train your own object detector, delete the following files (do not delete the folders):

•	All files in \object_detection\training
•	All files in \object_detection\inference_graph

Now, you are ready to start from scratch in training your own object detector. This tutorial will assume that all the files listed above were deleted, and will go on to explain how to generate the files for your own training dataset.

### 1d. Set up new Anaconda virtual environment ###

Next, we'll work on setting up a virtual environment in Anaconda for tensorflow-gpu. From the Start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”. If Windows asks you if you would like to allow it to make changes to your computer, click Yes.

In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command:
C:\> conda create -n tensorflow1 pip python=3.5

Then, activate the environment by issuing:

C:\> activate tensorflow1

Install tensorflow-gpu in this environment by issuing:

(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu

Install the other necessary packages by issuing the following commands:

(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
(Note: The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow, but they are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.)

### 1e. Configure PYTHONPATH environment variable ###

A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Do this by issuing the following commands (from any directory):
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim

(Note: Every time the "tensorflow1" virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again.)

### 1f. Compile Protobufs and run setup.py ###

Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. Every .proto file in the \object_detection\protos directory must be called out individually by the command.

In the Anaconda Command Prompt, change directories to the \models\research directory and copy and paste the following command into the command line and press Enter:

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto

This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.

Finally, run the following commands from the C:\tensorflow1\models\research directory:

(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install

### 1g. Test TensorFlow setup to verify it works ###

The TensorFlow Object Detection API is now all set up to use pre-trained models for object detection, or to train a new one. You can test it out and verify your installation is working by launching the object_detection_tutorial.ipynb script with Jupyter. From the \object_detection directory, issue this command:

(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
This opens the script in your default web browser and allows you to step through the code one section at a time. You can step through each section by clicking the “Run” button in the upper toolbar. The section is done running when the “In [ * ]” text next to the section populates with a number (e.g. “In [1]”).
Once you have stepped all the way through the script, you should see two labelled images at the bottom section the page. If you see this, then everything is working properly! If not, the bottom section will report any errors encountered.

### 2. Gather and Label Pictures ###

### 2a. Gather Pictures ###

TensorFlow needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have random objects in the image along with the desired objects, and should have a variety of backgrounds and lighting conditions. There should be some images where the desired object is partially obscured, overlapped with something else, or only halfway in the picture.

We have two different objects I want to detect (red, green). Then, it took about another 169 pictures with multiple images in the picture. I know I want to be able to detect the cards when they’re overlapping, so I made sure to have the cards be overlapped in many images.

![8](https://user-images.githubusercontent.com/24918359/56953180-8f2f1680-6b59-11e9-8df3-c8369b8fac26.png)

Make sure the images aren’t too large. They should be less than 200KB each, and their resolution shouldn’t be more than 720x1280. The larger the images are, the longer it will take to train the classifier. You can use the resizer.py script in this repository to reduce the size of the images.

![9](https://user-images.githubusercontent.com/24918359/56953182-8f2f1680-6b59-11e9-8446-305b2d82e0d2.png)

After you have all the pictures you need, move 20% of them to the \object_detection\images\test directory, and 80% of them to the \object_detection\images\train directory. Make sure there are a variety of pictures in both the \test and \train directories.

### 2b. Label Pictures ###

With all the pictures gathered, it’s time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it.

Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory. 

![10](https://user-images.githubusercontent.com/24918359/56953184-8f2f1680-6b59-11e9-8184-7ec582bca87e.png)

LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.

![11](https://user-images.githubusercontent.com/24918359/56953163-8c342600-6b59-11e9-9ab1-8d7a5ac1bbc0.png)

Also, can check if the size of each bounding box is correct by running sizeChecker.py

(tensorflow1) C:\tensorflow1\models\research\object_detection> python sizeChecker.py --move

![12](https://user-images.githubusercontent.com/24918359/56953164-8cccbc80-6b59-11e9-9080-308976164e9c.png)

### 3. Generate Training Data ###

With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. This tutorial uses the xml_to_csv.py and generate_tfrecord.py scripts from Dat Tran’s Raccoon Detector dataset, with some slight modifications to work with our directory structure.
First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.
Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file.

def class_text_to_int(row_label):
    if row_label == 'red':
        return 1
    elif row_label == 'green':
        return 2
    else:
        return None

For example:

def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        return None

Then, generate the TFRecord files by issuing these commands from the \object_detection folder:
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

###4. Create Label Map and Configure Training ###

The last thing to do before training is to create a label map and edit the training configuration file.

###4a. Label map ###

The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. 
item {
  id: 1
  name: 'red'
}

item {
  id: 2
  name: 'green'
}
}

The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the basketball, shirt, and shoe detector example mentioned, the labelmap.pbtxt file will look like:

item {
  id: 1
  name: 'basketball'
}

item {
  id: 2
  name: 'shirt'
}

item {
  id: 3
  name: 'shoe'
}

### 4b. Configure training ###

Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!
 
Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_pets.config file into the \object_detection\training directory

![13](https://user-images.githubusercontent.com/24918359/56953165-8cccbc80-6b59-11e9-82c8-056788bb04d2.png)

![14](https://user-images.githubusercontent.com/24918359/56953166-8cccbc80-6b59-11e9-8d61-bc1a36e04202.jpg)

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

### 5. Run the Training ###

Here we go! From the \object_detection directory, issue the following command to begin training:
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_signals.config

Each step of training reports the loss. It will start high and get lower and lower as training progresses. For my training on the Faster-RCNN-Inception-V2 model, it started at about 3.0 and quickly dropped below 0.8.

![15](https://user-images.githubusercontent.com/24918359/56953168-8d655300-6b59-11e9-8ee0-d3f22e9ddf33.png)

![16](https://user-images.githubusercontent.com/24918359/56953169-8d655300-6b59-11e9-8c76-9e9da0d354dc.png)


You can view the progress of the training job by using TensorBoard. To do this, open a new instance of Anaconda Prompt, activate the tensorflow1 virtual environment, change to the C:\tensorflow1\models\research\object_detection directory, and issue the following command:
(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training

![17](https://user-images.githubusercontent.com/24918359/56953170-8d655300-6b59-11e9-97f0-b71c18cbf2db.png)

The training routine periodically saves checkpoints about every five minutes. You can terminate the training by pressing Ctrl+C while in the command prompt window. I typically wait until just after a checkpoint has been saved to terminate the training. You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.

###6. Export Inference Graph ###

Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-1962” should be replaced with the highest-numbered .ckpt file in the training folder:

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v2_signal.config --trained_checkpoint_prefix training/model.ckpt-1962 --output_directory inference_graph

![18](https://user-images.githubusercontent.com/24918359/56953171-8d655300-6b59-11e9-95d1-5ae14ae023c6.png)

This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the object detection classifier.

###7. Use Newly Trained Object Detection Classifier ###

Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. So NUM_CLASSES = 2. To test your object detector, move a picture of the object or objects into the \object_detection folder, and change the IMAGE_NAME variable in the Object_detection_image.py to match the file name of the picture. Alternatively, you can use a video of the objects (using Object_detection_video.py), or just plug in a USB webcam and point it at the objects (using Object_detection_webcam.py).

