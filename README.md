# MazeVil AI

Welcome to the MazeVil AI project! This repository is dedicated to developing an AI agent capable of playing the game MazeVil autonomously. Our approach combines advanced machine learning techniques with sophisticated pathfinding and optimization algorithms to navigate the game environment and engage in combat sequences.

**Components**
- *TensorFlow Lite Models:* Lightweight neural networks optimized for real-time object detection.
- *YOLOv7 Models:* Cutting-edge object detection models known for their speed and accuracy.
- *SLAM Algorithms:* Techniques for creating a map of the game environment and localizing the player within it.
- *Optimization Algorithms:* Methods for enhancing the AI's decision-making during combat sequences.

A free demo version of the **Mazevil** game can be found at [Itch.io](https://splix.itch.io/mazevil).
The aim is to detect the path and the objects on the game screen and creating an algorithm to get the best score.

*Paths in [Test Scripts](Test Scripts) may need to be updated.*

## Table of Contents
- [Contributing](#contributing)
- [Data Preparation](#data-preparation)
  - [Data Gathering](#data-gathering)
  - [Labeling](#labeling)
- [Algorithms](#algorithms)
- [Traversing](#traversing)
- [Mapping](#mapping)
- [Fighting](#fighting)
- [YOLO Model](#yolo-model)
- [TF Lite Model](#tf-lite-model)
  - [Path and Object Detection](#path-and-object-detection)
    - [Custom Object Detection Test Model](#custom-object-detection-test-model)
    - [Path Detection System](#path-detection-system)
  - [Installation for Windows and TF GPU](#installation-for-windows-and-tf-gpu)
    - [1. Cuda Toolkit and cuDNN](#1-cuda-toolkit-and-cudnn)
    - [2. Anaconda Environment](#2-anaconda-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Install TF-Lite API](#4-install-tf-lite-api)
- [License](#license)

## Contributing
We welcome contributions from the community! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Data Preparation

### Data Gathering

### Labeling

## Algorithms

### Traversing

### Mapping

### Fighting

## YOLO Model

Coming Soon...

## TF Lite Model

### Path and Object Detection

#### Custom Object Detection Test Model

The trained script is taken from [colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb). See [here](https://www.youtube.com/watch?v=XZ7FYAMCc4M&t=311s) for the Youtube Video.

The data is taken using [LabelImg](https://github.com/HumanSignal/labelImg).
A [test model](Model\test_model_001) is trained with 183 images. Images and XML files can be found at [**Data**](Data).

**Examples of [Test Model 001]((Model\test_model_001)) with a score threshold of 0.4:**

<p align="center">
  <img src="Docs\object_2.png" alt="object_2" width="400"/>
  <img src="Docs\object_0 .png" alt="object_0" width="400"/>
</p>

#### Path Detection

**Examples of Path Detection:**

<p align="center">
  <img src="Docs\path_0.png" alt="path_0" width="400"/>
  <img src="Docs\path_1.png" alt="path_1" width="400"/>
</p>

### Installation for Windows and TF Lite

#### 1. Cuda Toolkit and cuDNN

##### 1.1 Cuda Installation

Download and install version 11.3 of [Cuda Toolkit](https://developer.nvidia.com/cuda-11.3.0-download-archive) from [https://developer.nvidia.com/cuda-11.3.0-download-archive](https://developer.nvidia.com/cuda-11.3.0-download-archive).

##### 1.2 cuDNN Installation

Download cuDNN v8.2.0 (April 23rd, 2021), for CUDA 11.x from [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive).

 - Copy the contents of the bin folder from the cuDNN archive to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin** directory.
 - Copy the contents of the included folder to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include** directory.
 - Copy the contents of the lib folder to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64** directory.

##### 1.3 Update Environment Variables

 - Open the Start Menu and search for "Environment Variables."
 - Click "Edit the system environment variables."
 - Click the "Environment Variables" button in the System Properties window.
 - In the Environment Variables window, find the Path variable under "System variables," select it, and click "Edit."
 - Add **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin** to the list if it’s not already there.

#### 2. Anaconda Environment

Create an Anaconda venv with the *python version 3.9*:
```
    conda create -n {envname} python=3.9
```
Activate environment:
```
    conda activate {envname}
```

#### 3. Install Dependicies

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

#### 4. Install TF-Lite API

Installation is taken from [https://www.youtube.com/watch?v=rRwflsS67ow&t=780s](https://www.youtube.com/watch?v=rRwflsS67ow&t=780s)

##### 4.1 Tensorflow Models Repository

Download [Tensorflow Models Repo](https://github.com/tensorflow/models.git) to your created file. Let's say the file is path/TF2.
In Command Prompt, git can be used to download: 
```
    C:\path\TF2> git clone https://github.com/tensorflow/models.git
```

##### 4.2 Protobuf Repository

Go to the [Protobuf Repository Releases](https://github.com/protocolbuffers/protobuf/releases) and find the release *version 27.3*.
Download *protoc-27.3-win64.zip* file.
Unzip the folder.
Add the bin folder (**C:/pathto/protoc-27.3-win64/bin/**) to the environment paths.

##### 4.3 Protoc

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
Expect no output and no errors.

##### 4.4 Setup

Copy the setup.py file from **/path/TF2/models/research/object_detection/packages/tf2/** to **/path/TF2/models/research/**
Open Command Prompt with the path /path/TF2/models/research and type:
```
    python -m pip install .
```

Reinstall numpy using conda:
```
    conda install numpy
```

##### 4.5 Test Installation
Open Command Prompt with the path /path/TF2/models/research and type:
```
    python objet_detection/builders/model_builder_tf2_test.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.