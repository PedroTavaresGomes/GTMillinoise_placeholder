# _GT-MilliNoise_: Graph Transformer for Point-wise Denoising of Indoor Millimeter-Wave Point Clouds

(Paper under submission) [[Project]]() [[Paper]]() 

This GitHub contains the code to implement  _GT-MilliNoise_ architecture for millimeter-wave point cloud denoising.
The architecture was designed using the _MilliNoise_ dataset.    

MillNoise Dataset: [[Dataset]](https://github.com/c3lab/MilliNoise)  [[Paper]](https://dl.acm.org/doi/10.1145/3625468.3652189) 


## Overview
<img src="https://github.com/PedroTavaresGomes/GTMillinoise_placeholder/blob/main/imgs/pipeline.png" scale="0.2">


## Citation
Please cite this paper if you want to use it in your work,

	@article{coming soon,
	  }
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.10.0, CUDA 9.0 and cuDNN 7.3

Compile the code. You must select the correct CUDA version and install Tensorflow on your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`


## Directory paths
--log-dir

--data-dir

### Usage 
#### 
To train a model for millimeter-wave point cloud denoising:

    python train.py --version <name_of_model> --data-split <#split> --model <architecture_name>  --seq-length <input_frames>

For example:

    python train.py --version v0 --data-split 4 --model TG  --seq-length 12 

Trains a GT-Millinoise model using dataset split #4 (Fold-4)

To evaluate the model

    python test.py --version v0 --data-split 4 --model TG --seq-length 12 --manual-restore 2 (loads best model in validation)
#### Splits:
| #Split: | K-Fold: | Training Scenarios | Test-Scenarios    
|---|---|---|---|
| --data-split 11  | Fold 1-6 | All [1-6] | All [1-6] |
| --data-split 17  | Fold 1,2,3 | [4,5,6] | [1,2,3] |
| --data-split 4  | Fold 4 | [1,2,3,5,6] | [4] |
| --data-split 16  | Fold 5 | [1,2,3,4,6] | [5] |

#### Available Models:
| Model | Name | Description |     
|---|---|---|
| --model TG  | GT-Millinoise | Standart GT-Millinoise architecture |
| --model PointNet  | PoinNet | |
| --model PointNet_2  | PointNet++ |  |
| --model DGCNN  | Dynamci Graph CNN (DGCNN) |  |
| --model Transformer  | Vannila-Transformer |  |
| --model GT_intensity  | GT-Millinoise | GT-Millinoise with intensity as input |
| --model GT_velocity  | GT-Millinoise | GT-Millinoise with velocity as input |
| --model GT_noTC  | GT-Millinoise | GT-Millinoise with  without temporal block |



### Datasets
The models were evaluated with the following datasets:
1. [Moving MNIST Point Cloud (1 digit)](https://drive.google.com/open?id=17RpNwMLDcR5fLr0DJkRxmC5WgFn3RwK_) &emsp; 2. [Moving MNIST Point Cloud (2 digits)](https://drive.google.com/open?id=11EkVsE5fmgU5D5GsOATQ6XN17gmn7IvF) &emsp; 3. [JPEG Dynamic Human Bodies (4000 points)](https://drive.google.com/file/d/1hbB1EPKq3UVlXUL5m81M1E6_s5lWmoB-/view)

To create the Human Bodies dataset, follow the instructions in the Dataset folder.

## Visual Results

![with = 0.25/pagewith](gif_results_fast.gif)

## Acknowledgement
The parts of this codebase are borrowed from Related Repos:

### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn
4. Millinoise https://github.com/c3lab/MilliNoise

