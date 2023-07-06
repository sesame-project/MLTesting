# GenerativeRepair: Semantic Data Augmentation Methodology for Deep Learning Hardening and Repairing using Generative AI
###### This is the SESAME Hardening and Repairing Framework of Data-Driven and Learning Components of EDDIs

The code for our framework GenerativeRepair is available to execution using two main scipts repairing.py and fuzzing.py.
It is in beta version and provided as a reference to the approaches and methodologies presented in deliverable 6.4. 
If you encounter any problems, do not hesitate to reach out to the York team (sondess.missaoui@york.ac.uk).

Scripts are tested with the open-source machine learning framework Keras (v2.2.2) with Tensorflow (v2.6) backend.

## Abstract
Our framework is a design-time artifact, it focuses on improving and hardening the EDDI capability for fault diagnosis.

The framework has a twofold objective: 1) monitor the Data-Driven and Learning components (Deep Learning models) that are part of the MRS (e.g. person detection or object recognition tasks), and 2) improve the Deep Learning model's robustness in the MRS system.

It encompasses two tools:
### -GenerativeFuzzer: Hardening tool to improve the DL testing process: 
Automatically generate corner cases that can lead to unsafe behaviour, i.e., misclassification using semantic data augmentation methods. The synthetic data serve as a vulnerability discovery technology to assess the safety of the model at design time, thus reducing the risk of potentially hazardous behaviour at runtime.

### -SafetyRepair: Repairing the DL model:
Tackling the problem of correcting the DNN model once unsafe behaviour is found. The tool deploys the synthetic corner cases to retrain the DNN model. This helps the DNN model to learn more robust and generalisable representations of the data distribution, leading to better performance on unseen data samples, which could incur failure cases (i.e., misclassification or imprecise detection) in the operational environment.
![alt text](https://github.com/sesame-project/MLTesting/blob/Knw/results/archi.png)
```
![alt text](http://url/to/img.png)
```
## Installation Requirement Packages
We recommend starting by creating a virtual environment and then installing the required packages.

### Virtual Environement

#### Linux
```
python3 -m pip install --user virtualenv

python3 -m venv path/to/the/virtual/environment
```
###### Activate virtual environment

```
source path/to/the/virtual/environment/bin/activate
```
####  conda 

1- Open up the Anaconda command prompt.
2- type:
```
conda create -n path/to/the/virtual/environment python=x.x anaconda

```
specify your Python version by replacing the `python=x.x' in the previous command line.

###### Activate virtual environment
```
conda activate path/to/the/virtual/environment
```
### Required packages

Instead of installing packages individually,  we provide you with the requirements file and pip allows you to declare all dependencies as follows:

```
 pip install -r GenerativeRepair_requirements.txt

```
you need also to install Stable Diffusion model from Hugging Face as follow:
```
pip install https://github.com/huggingface/diffusers/archive/main.zip -qUU --ignore-installed

pip install transformers -q -UU ftfy gradio
```
you need to create a Hugging Face account and get token to download and install Stable Diffusion Inpainting


```
pip freeze | grep diffusers

wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/inference/inpainting.py

```



## Running GenerativeRepair
For Fuzz testing GenerativeFuzzer, run fuzzing.py script using the shell command

```
$ cd path/to/the/project/folder

$ Python3.8 fuzzing.py -method[0 or 1] -fuzzer[type_of_fuzzing]  -repair[continuous_learning_paradigm] -it[nbre_of_iteration] -model [path_to_keras_model_file] -dataset [dataset_name] -approach [coverage_criteria] -logfile [path_to_log_file]

```
For model re-training SafetyRepair, run repairing.py script using the shell command

```
$ cd path/to/the/project/folder

$ Python3.8 repairing.py -method[0 or 1] -fuzzer[type_of_fuzzing]  -repair[continuous_learning_paradigm] -it[nbre_of_iteration] -model [path_to_keras_model_file] -dataset [dataset_name] -approach [coverage_criteria] -logfile [path_to_log_file]

```

## Parameters for configuring 
```
-method : an integer took 0 for fuzzing operation and 1 for repairing.

-fuzzer : the name of the selected fuzzer. Our platform provides 5 strategies, with the possibility of choosing a combination of different testing criteria as guidance. These fuzzers are:

  Random Noise testing (RN). This combines random sampling as seed selection strategy and Gaussian Noise for data augmentation
  Random Inpainting (RInp). This fuzzer uses random sampling strategy and test-guided Stable Diffusion Inpainting as data augmentation.
  Random Semantic Occlusion (SemOcc). This fuzzer uses random sampling strategy with  Semantic occlusion (both random erasing and   synthetic occlusion similarly) to augment each input seed.
  DeepKnowledge Inpainting (KnwInp). Different from RInp, this strategy guides testing using DeepKnowledge coverage criteria as feedback. An input seed is put to the seed queue if it improves the Deepknowledge coverage.

-repair : this the selected paradigm for continuous learning. We can select :
CLTask : Task incremental learning
CLClass : Class incremental learning 
CL : Continuous learning of known classes

-it : number of iterations for data augmentation within the fuzzing process. We advise to select an integer between 2 an 10.

-app: for approach. The approach for coverage estimation. The selected coverage is used as guidance in each iteration to pick the augmented seed as new test. Our current implementation supports DeepKnowledge (Knw), DeepImportance (idc),(nc),(kmnc),(nbc),(snac),(tknc),(ssc), (lsa), and (dsa).

-model : the name of the Keras model file. The trained keras model is saved as .hdf5 file or the architecture can be saved as JSON and the weightssaved separately as an .h5 file. All the trained model are saved under the folder `Networks’. Our implementation provides three trained DNN models including Allconvnet.h5, LeNet5.h5, and Vgg19.h5.

-dataset : name of the dataset to be used. Current implementation supports Cifar-10 (cifar) COCO (coco) and grape leaves (grape). Our platform is extensible and other dataset can be added by modifying `Dataprocessing.py’ and `Data_Augment.py’ scripts.

-layer : The subject layer’s index for approaches including ‘idc’,‘tknc’, ‘lsa’. Note that only trainablelayers can be selected.

- logfile : The name of the file that the results are to be saved.


```
### Results 
