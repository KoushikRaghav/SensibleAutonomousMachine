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





