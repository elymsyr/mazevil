# MAZEVIL

A free demo version for **Mazevil** game can be found at [Itch.io](https://splix.itch.io/mazevil).
This project is focused on custom object detection using Tensorflow API and path detection with OpenCV.
The aim is to detect the path and the object on the game screen and train a model using Reinforcement Learning algorithms.

## Table of Contents
- [Path and Object Detection](#path-and-object-detection)
  - [Custom Object Detection Test Model](#custom-object-detection-test-model)
- [Path Detection System](#path-detection-system)
- [Installation](#installation)
  - [Installation for Windows and TF GPU](#installation-for-windows-and-tf-gpu)
  - [1. Tensorflow Models Repository](#1-tensorflow-models-repository)
  - [2. Protobuf Repository](#2-protobuf-repository)
  - [3. Cuda Toolkit and cuDNN](#3-cuda-toolkit-and-cudnn)
      - [3.1 Cuda Installation](#31-cuda-installation)
      - [3.2 cuDNN Installation](#32-cudnn-installation)
      - [3.3 Update Environment Variables](#33-update-environment-variables)
  - [4. Anaconda Environment](#4-anaconda-environment)
  - [5. Install Dependencies](#5-install-dependencies)
  - [6. Install TF-Lite API](#6-install-tf-lite-api)
      - [6.1 Protoc](#61-protoc)
      - [6.2 Setup](#62-setup)
      - [6.3 Test Installation](#63-test-installation)

## Path and Object Detection

### Custom Object Detection Test Model

Trained script is taken from [colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb). See [here](https://www.youtube.com/watch?v=XZ7FYAMCc4M&t=311s) for the Youtube Video.

The data is taken using [LabelImg](https://github.com/HumanSignal/labelImg).
A [test model](Model\test_model_001) is trained with 183 images. Images and xml files can be found at [**Data**](Data).

**Examples of [Test Model 001]((Model\test_model_001)) with a score threshold of 0.4:**

<p align="center">
  <img src="Docs\object_2.png" alt="object_2" width="400"/>
  <img src="Docs\object_0 .png" alt="object_0" width="400"/>
</p>

## Path Detection System

**Examples of Path Detection:**

<p align="center">
  <img src="Docs\path_0 .png" alt="path_0" width="400"/>
  <img src="Docs\path_1.png" alt="path_1" width="400"/>
</p>

## Installation

### Installation for Windows and YOLO

Coming soon :)

### Installation for Windows and TF GPU

#### 1. Tensorflow Models Repository

Download [Tensorflow Models Repo](https://github.com/tensorflow/models.git) to a file you have created. Let's say the file is path/TF2.
In Command Prompt, git can be used to download: 
```
    C:\path\TF2> git clone https://github.com/tensorflow/models.git
```

#### 2. Protobuf Repository

Go to the [Protobuf Repository Releases](https://github.com/protocolbuffers/protobuf/releases) and find the release *version 27.3*.
Download *protoc-27.3-win64.zip* file.
Unzip the folder.
Add the bin folder (**C:/pathto/protoc-27.3-win64/bin/**) to the environment paths.

#### 3. Cuda Toolkit and cuDNN

##### 3.1 Cuda Installation

Download and install verion 11.3 of [Cuda Toolkit](https://developer.nvidia.com/cuda-11.3.0-download-archive) from [https://developer.nvidia.com/cuda-11.3.0-download-archive](https://developer.nvidia.com/cuda-11.3.0-download-archive).

##### 3.2 cuDNN Installation

Download cuDNN v8.2.0 (April 23rd, 2021), for CUDA 11.x from [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive).

 - Copy the contents of the bin folder from the cuDNN archive to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin** directory.
 - Copy the contents of the include folder to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include** directory.
 - Copy the contents of the lib folder to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64** directory.

##### 3.3 Update Environment Variables

 - Open the Start Menu and search for "Environment Variables."
 - Click "Edit the system environment variables."
 - In the System Properties window, click the "Environment Variables" button.
 - In the Environment Variables window, find the Path variable under "System variables," select it, and click "Edit."
 - Add **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin** to the list if it’s not already there.

#### 4. Anaconda Environment

Create an Anaconda venv with the *python version 3.9*:
```
    conda create -n {envname} python=3.9
```
Activate environment:
```
    conda activate {envname}
```

#### 5. Install Dependicies

```
    conda install cudatoolkit=11.0 cudnn=8.2
```
**or**
```
    conda install cudatoolkit=11.0 cudnn=8.0
```
**and**
```
    pip install protobuf==3.19.6 Cython==3.0.10 httplib2==0.21.0 opencv-python==4.10 nvidia-cudnn-cu11==9.3 pandas==1.1.5 protobuf==3.19.6 PyAutoGUI==0.9.54 PyGetWindow==0.0.9 scikit-learn==1.0.2 scipy==1.7.3 tensorflow-gpu==2.4.0 urllib3==1.26.6 PyYAML==5.4.1
```

#### 6. Install TF-Lite API

Installation is taken from [https://www.youtube.com/watch?v=rRwflsS67ow&t=780s](https://www.youtube.com/watch?v=rRwflsS67ow&t=780s)

##### 6.1 Protoc

Open Command Prompt with the path /path/TF2/models/research.
Create a .py file with the following code and the name ***use_protobuf.py***:
```
    import sys
    import os
    args = sys.argv
    directory = args[1]
    protoc_path = args[2]
    for file in os.listdir(directory):
        if file.endswith(".proto):
            os.systems(protoc_path+" "+directory+"/"+"file+" __python_out=.")
```
Locate file to the /path/TF2/models/research.
Run script from Command Prompt:
```
    python use_protobuf.py object_detection/protos protoc
```
Expect no ouput and no errors.

##### 6.2 Setup

Copy the setup.py file from **/path/TF2/models/research/object_detection/packages/tf2/** to **/path/TF2/models/research/**
Open Command Prompt with the path /path/TF2/models/research and type:
```
    python -m pip install .
```

Reinstall numpy using conda:
```
    conda install numpy
```

##### 6.3 Test Installation
Open Command Prompt with the path /path/TF2/models/research and type:
```
    python objet_detection/builders/model_builder_tf2_test.py
```