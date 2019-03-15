# FirstML
Implementation of Imagetop training for newcomers:

### Set up
#### SSH into HEXDL
ssh into hexcms and then:
```
ssh -X hexcmsusername@HEXDL
```
enter hexcms password

#### Required python3 modules (if not installed)
```
pip3 install numpy
pip3 install scipy
pip3 install pandas
```

#### Checkout from github (somewhere with some space)
```
git clone https://github.com/knash/FirstML.git
```

#### Run test program
type
```
nvidia-smi
```
to see if any gpus are in use.  You should see something like
```
HEXDL:~/FirstML> nvidia-smi
Mon Feb 18 09:35:33 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 396.26                 Driver Version: 396.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
| 38%   51C    P8    19W / 250W |     74MiB / 12192MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  TITAN Xp            Off  | 00000000:02:00.0 Off |                  N/A |
| 23%   27C    P8    16W / 250W |      2MiB / 12196MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1673      G   /usr/lib/xorg/Xorg                            71MiB |
+-----------------------------------------------------------------------------+
```
Here, no gpu is in use, so in the following command you would use either -g 0 or -g 1 to choose the gpu to run on.

```
cd FirstML
python3 MLstarter.py -s SIGforPhoAll.dat -b QCDconstpt.dat -c 0,1,2,3,4,5 -d 1,2,3,4,5,6,7,8,9,10,11,12 -p testing -f 0.2 -e 2 -g 1
```
Note that for a real training use -f 1.0 and -e 1000

While this is running, the gpu 1 will start running, which changes the nvidia-smi section to look something like

```
+-------------------------------+----------------------+----------------------+
|   1  TITAN Xp            Off  | 00000000:02:00.0 Off |                  N/A |
| 23%   38C    P2   227W / 250W |  11317MiB / 12196MiB |     84%      Default |
+-------------------------------+----------------------+----------------------+
```

As the script runs, the categorial accuracy should improve toward convergence with the final
test accuracy being a good measure of sensitivity.  Also, the script prints the area under the ROC curve ("AUC") which can be used as a
measure of sensitivity as well

After the script runs, there will be two new files in the weights/ directory. The weights_*name*.hdf5 and model_*name*.h5.
The weights_*name*.hdf5 file can be loaded back into keras for classification (see the "-l" option below).  The model_*name*.h5 file
has both the weights and model architecture and can be loaded into bare tensorflow (and therefore used with CMSSW).
Also output is a file in the analysis/ROC folder with the ROC curve X and Y coordinates in CSV format.

#### Import into CMSSW
The format used for loading models into tensorflow is protobuf (.pb).  For this we use a private repository aptly named keras_to_tensorflow.
```
git clone https://github.com/amir-abdi/keras_to_tensorflow.git
cd keras-to-tensorflow
eval python3 k2tf_convert.py -m ../model_*name*.h5 -n 1
```

#### ascii file creation
These are created through crab.  See the description here:
https://github.com/knash/NanoHRT/tree/ForTraining


#### Description
MLstarter.py is a starter machine learning program based off of a modified version of https://arxiv.org/abs/1803.00107.
This program trains and tests but performs none of the preprocessing (center,normalize,rotate).
The input files are ascii and are in the following format [[[X,Y],[C0,C1,...CN],...],truth(1,0),dense0,dense1,...,denseN] where [C0,C1,...CN] are the colors, [X,Y] are the pixel coordinates (0-37) and dense0,dense1,...,denseN are the inputs to the dense layer.  
The pixelized images should be normalized to unit intensity (ie pt).  Also, if there is any order to the images (ie running over multiple signal sets in order) then the ascii files should be shuffled.

#### Options
```
'-s', '--signal': The input signal filename (without path)
'-b', '--background': The input background filename (without path)
'-m', '--mode': The mode to run in (train or test)
'-l', '--load': Pre-trained weights file to load (for continuing training or testing without retraining)
'-c', '--colors': A csv list of the indices of the colors (in N) to use
'-d', '--dense': A csv list of the indices of the dense inputs (in denseN) to use
'-p', '--post': A string to append to the output filenames created by the script
'-g', '--gpus': A csv list of the GPUs to use (for HEXDL this is 0 and 1).  Before running, use nvidia-smi to see if a GPU is in use.
If it is, then use the other one.  Can also specify 0,1 to use both, but this is still in testing.
'-D', '--directory': Path used by the -s and -b options
'-e', '--epochs': Total number of training epochs.  After this is reached, prematurely end the training.
If this is set to a high number (1000) then the training ends after the convergence criterea are met.
'-f', '--fraction': The fraction of the total input events to use (set low for a test run).
'--skipgen': Enter this option to skip the train,test,validate set generation (ie if it is already created)
```
