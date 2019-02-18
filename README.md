# FirstML
Implementation of Deeptop training for newcomers:

### Set up
#### SSH into HEXDL
ssh into hexcms and then:
```
ssh -X hexcmsusername@HEXDL
```
enter hexcms password

#### Checkout from github
```
git clone https://github.com/knash/FirstML.git
```
#### Run test program
```
python3 MLstarter.py -s SIGforPhoAll.dat -b QCDconstpt.dat -c 0,1,2,3,4,5 -d 1,2,3,4,5,6,7,8,9,10,11,12 -p test -f 0.2
```

#### Description
MLstarter.py is a starter machine learning program based off of a modified version of https://arxiv.org/abs/1803.00107self.
This program trains and tests but performs none of the preprocessing (center,normalize,rotate).
The input files are ascii and are in the following format [[37x37xN],truth(1,0),dense0,dense1,...,denseN] where N is the number of colors and dense0...
are the inputs to the dense layer.

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
