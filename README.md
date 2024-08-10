# MazeVil AI


<p align="center">
  <img src="Docs\main.png" alt="object_2" width="400"/>
</p>

Welcome to the MazeVil AI project! This repository is dedicated to developing an AI agent capable of playing the game MazeVil autonomously. My approach combines machine learning techniques with pathfinding and optimization algorithms to navigate the game environment and engage in combat sequences. The aim is to detect the path and the objects on the game screen and create an algorithm to get the best score, maybe even kill the boss.

A free demo version of the **Mazevil** game can be found at [Itch.io](https://splix.itch.io/mazevil).

*Paths in [Test Scripts](Test%20Scripts) may need to be updated.*

## Latest Updates

- **Commit (3fa2674):** All the files and folders of yolov7 model is now moved from [YOLO Model](YOLO%20Model) into the folder [yolov7](Test%20Scripts\yolov7). Paths may need to be updated.

<!-- - **Release:** v1.2.3 (Released on 2024-08-10)
- **Commit:** 1a2b3c4d5e6f7g8h9i0j (2024-08-10) -->

For detailed changes, see the [release notes](https://github.com/yourusername/yourrepo/releases/) or the [commit history](https://github.com/yourusername/yourrepo/commits/main).

## Table of Contents
- [Contributing](#contributing)
- [Data Preparation](#data-preparation)
  - [Data Gathering](#data-gathering)
  - [Labeling](#labeling)
- [Algorithms](#algorithms)
  - [Traversing](#traversing)
  - [Mapping](#mapping)
  - [Fighting](#fighting)
- [Yolov8](#yolov8)
- [TF Lite Model](#tf-lite-model)
  - [Object Detection](#object-detection-tf)
  - [Installation of TF Lite for Windows](#installation-of-tf-lite-for-windows)
    - [1. Cuda Toolkit and cuDNN](#1-cuda-toolkit-and-cudnn)
    - [2. Anaconda Environment](#2-anaconda-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Install TF-Lite API](#4-install-tf-lite-api)
- [Yolov7](#yolov7)
- [Helpers](#helpers)
- [License](#license)

## Contributing
We welcome contributions from the community! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change. See [*CONTRIBUTING.md*](CONTRIBUTING.md) for more...

## Data Preparation

### Data Gathering

The data acquisition process involves systematically capturing screenshots from the game environment using two specialized scripts [*ss.py*](TF%20Model\Data\ss.py). The script is meticulously designed to automate the screenshot capturing process, ensuring that the data reflects a diverse range of in-game scenarios. This data is then utilized to train and refine the models, making them adept at recognizing and interpreting various game elements with high accuracy.

### Labeling

The labeling of the captured data is a critical step in preparing it for machine learning applications. This process is carried out using the [**LabelImg**](https://github.com/HumanSignal/labelImg) application, a robust tool for annotating images with precise bounding boxes. By accurately labeling each object within the game screenshots, we ensure that the model can effectively learn to differentiate between various elements, such as enemies, obstacles, and key items. The quality of this labeling process directly impacts the performance and reliability of the trained models.

The labels of the models can be found in the *Model* folders in [*TF Model*](TF%20Model) or [*YOLO Model*](YOLO%20Model).

## Algorithms

The following sections outline a conceptual plan for the project, detailing the envisioned approach to implementation of various algorithms. While these descriptions reflect the intended methodology and design, they remain preliminary and subject to change as development progresses. 

### Traversing

The traversal of the game environment is guided by the Greedy Best-First Search Algorithm, chosen for its optimal balance between speed and computational efficiency. As the player navigates through the dungeon rooms, the algorithm dynamically searches for the shortest path to the key, taking into account the layout and obstacles within each room. This algorithm excels at identifying the most promising path by prioritizing moves that seem to bring the player closer to the goal. The ultimate aim is to ensure a seamless and efficient exploration process, where the player is consistently directed toward the next closest dungeon room.

### Mapping

To effectively navigate and visit different dungeon rooms, the system must maintain an updated map of the discovered rooms and pathways. The script [path.py](TF%20Model\Project\path.py) is integral to this mapping process, continuously analyzing the screen to track the player's progress. While inactive traps are marked as safe paths, the script updates the current screen's data, which is stored in an array. As the player moves through the game, both the values and dimensions of another array—initialized at the start of the game—are updated to reflect the changing environment. Although a basic SLAM (Simultaneous Localization and Mapping) algorithm could be implemented to enhance this mapping, the project is currently in the early stages of development, with more advanced features yet to be realized.

**Examples of Path Detection (An Early Version):**

<p align="center">
  <img src="Docs\path_0.png" alt="path_0" width="400"/>
  <img src="Docs\path_1.png" alt="path_1" width="400"/>
</p>

### Fighting

The combat system is designed to adapt to different enemy types, focusing on dodging ranged attacks while maintaining a safe distance from melee attackers. The player is programmed to keep the mouse cursor trained on the nearest enemy, continuously holding down the left mouse button to attack. This straightforward combat approach not changes a lot when the player enters a boss dungeon room, however the challenges escalate. Although the current plan provides a solid foundation, there may be still many aspects of the combat system that need to be refined and expanded as the project progresses.

## Yolov8

### Installation

#### 1. Cuda and cuDNN

See [Insatallation for Cuda Toolkit vX.Y and cuDNN vA.B.C](#insatallation-for-cuda-toolkit-vx.y-and-cudnn-va.b.c). Foloow the instructions and isnstall CUDA 11.8 and cuDNN v8.9.7 (December 5th, 2023), for CUDA 11.x.

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
    conda install cudatoolkit=X.Y cudnn=x.x
```
**or**
```
    conda install cudatoolkit=X.Y cudnn=x.x
```
**and**
```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```
    pip install ultralytics
```


## TF Lite Model

### <a name="object-detection-tf">Object Detection</a>

**Custom Object Detection Test Model**

The train script is taken from [colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb). See [here](https://www.youtube.com/watch?v=XZ7FYAMCc4M&t=311s) for the Youtube Video.

The data is taken using [LabelImg](https://github.com/HumanSignal/labelImg).
The test model [test_0](TF%20Model\Model\test_0) is trained with 183 images. Images and XML files can be found at [**Data**](Data).

**Examples of the model [test_0](TF%20Model\Model\test_0) with a score threshold of 0.4:**

<p align="center">
  <img src="Docs\object_2.png" alt="object_2" width="400"/>
  <img src="Docs\object_0 .png" alt="object_0" width="400"/>
</p>

### Installation of TF Lite for Windows

#### 1. Cuda Toolkit and cuDNN

See [Insatallation for Cuda Toolkit vX.Y and cuDNN vA.B.C](#insatallation-for-cuda-toolkit-vx.y-and-cudnn-va.b.c). Foloow the instructions and isnstall CUDA 11.3 and cuDNN v8.2 or v8.0 (December 5th, 2023), for CUDA 11.x.

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



## Yolov7

I conducted a series of tests comparing YOLO and TensorFlow Lite (TF Lite) models using the same randomly mixed image data and labels. The objective was to evaluate both models' performance in terms of accuracy and real-time latency. Although YOLO showed better accuracy in object detection, I encountered significant challenges with system performance while using it. Specifically, the YOLO model caused my older desktop computer to slow down drastically, making it nearly impossible to obtain results in real-time.

*In addition, deploying models on a cloud service and obtaining detection results via an API will be considered and tested as a potential solution to address performance concerns and further optimize results.*


## Helpers

### Insatallation for Cuda Toolkit vX.Y and cuDNN vA.B.C

#### 1.1 Cuda Installation

Download and install [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) from [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive).

#### 1.2 cuDNN Installation

Download cuDNN vA.B.C, for CUDA X.x from [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive).

 - Copy the contents of the bin folder from the cuDNN archive to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin** directory.
 - Copy the contents of the included folder to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\include** directory.
 - Copy the contents of the lib/x64 folder to the **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\lib\x64** directory.

#### 1.3 Update Environment Variables

 - Open the Start Menu and search for "Environment Variables."
 - Click "Edit the system environment variables."
 - Click the "Environment Variables" button in the System Properties window.
 - Create a new variable under "System variables", with the following values:
  - Var name: *CUDNN*
  - Var value: *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\include;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\lib*

#### Test Installation

Run on python:
```
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```
Run on Command Prompt or Powershell:
```
C:\> nvcc --version
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
