#example: python3 MLstarter.py -s SIGforPhoAll.dat -b QCDconstpt.dat -c 0,1,2,3,4,5 -d 1,2,3,4,5,6,7,8,9,10,11,12 -p test -f 0.2
#LOAD LIBRARIES
##-------------------------------------------------------------------------------

from __future__ import print_function
import copy, os, sys, time, logging
import numpy as np
import json
np.random.seed(1560)
import keras
print('using keras version:',keras.__version__)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.utils import np_utils
from keras.utils import multi_gpu_model
import shuffle as shf
import tensorflow as tf
print('using tensorflow version: ',tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
import tensorflow as tf
from keras.backend import tensorflow_backend as K

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-s', '--signal', metavar='F', type='string', action='store',
                  default	=	'None',
                  dest		=	'signal',
                  help		=	'signal dat file locastion')
parser.add_option('-b', '--background', metavar='F', type='string', action='store',
                  default	=	'None',
                  dest		=	'background',
                  help		=	'background dat file locastion')
parser.add_option('-m', '--mode', metavar='F', type='string', action='store',
                  default	=	'train',
                  dest		=	'mode',
                  help		=	'what is being performed.  train or test')
parser.add_option('-l', '--load', metavar='F', type='string', action='store',
                  default	=	'None',
                  dest		=	'load',
                  help		=	'weights file to preload (.h5 format).  This lets you continue training or test pre-trained weights -- None starts fresh.')
parser.add_option('-c', '--colors', metavar='F', type='string', action='store',
                  default	=	'0,1',
                  dest		=	'colors',
                  help		=	'csv of color indices to process')
parser.add_option('-d', '--dense', metavar='F', type='string', action='store',
                  default	=	'1,2,3,4,5,6,7,8,9,10,11,12',
                  dest		=	'dense',
                  help		=	'csv of dense-layer indices to process')
parser.add_option('-p', '--post', metavar='F', type='string', action='store',
                  default	=	'',
                  dest		=	'post',
                  help		=	'string to append to created filenames')
parser.add_option('-g', '--gpus', metavar='F', type='string', action='store',
                  default	=	'0',
                  dest		=	'gpus',
                  help		=	'csv of gpus to run on -- 0 or 1 or 0,1')
parser.add_option('-D', '--directory', metavar='F', type='string', action='store',
                  default	=	'/localhome/knash/deeptop/JetImages/kevin/',
                  dest		=	'directory',
                  help		=	'Directory to look for signal+background dat files')
parser.add_option('-e', '--epochs', metavar='F', type='int', action='store',
                  default	=	1000,
                  dest		=	'epochs',
                  help		=	'total number of epochs')
parser.add_option('-f', '--fraction', metavar='F', type='float', action='store',
                  default	=	1.0,
                  dest		=	'fraction',
                  help		=	'fraction of images to process')
parser.add_option('--skipgen', metavar='F', action='store_true',
                  default=False,
                  dest='skipgen',
                  help='skip the train,test,validate set generation (ie if it is already created)')

(options, args) = parser.parse_args()

print('Options summary')
print('==================')
for opt,value in options.__dict__.items():
    print(str(opt) +': '+ str(value))
print('==================')



os.environ['CUDA_VISIBLE_DEVICES']=options.gpus
config.gpu_options.per_process_gpu_memory_fraction = 1.0

set_session(tf.Session(config=config))

start_time = time.time()

np.set_printoptions(threshold=np.nan)

##------------------------------------------------------------------------------
# Global variables
##------------------------------------------------------------------------------

image_array_dir_in=options.directory

signalfilename= options.signal
backgroundfilename=options.background

extrastringarray = signalfilename.split('_')
extrastring = extrastringarray[-1]

post = options.post
print('Input directory',image_array_dir_in)
name_sg=str('_'.join(signalfilename.split('_')[:2]))
name_bg=str('_'.join(backgroundfilename.split('_')[:2]))
print('Name signal ={}'.format(name_sg))
print('Name background ={}'.format(name_bg))
print('-----------'*10)


#This array refers to the color index (which is inside the first element of the ascii event)
colarray = options.colors.split(',')
for i in range(0,len(colarray)):
	colarray[i]=int(colarray[i])

#This array refers to the dense layer index (which starts at the third element of the ascii event)
densearray = options.dense.split(',')
for i in range(0,len(densearray)):
	densearray[i]=int(densearray[i])+2

gpuarray = options.gpus.split(',')
for i in range(0,len(gpuarray)):
	gpuarray[i]=int(gpuarray[i])
npoints = 38
img_rows, img_cols = npoints-1, npoints-1
N_pixels=np.power(npoints-1,2)
my_batch_size = 128
num_classes = 2
epochs =int(options.epochs)
sample_relative_size=float(options.fraction)
mode=options.mode

ncolors=len(colarray)
ndense=len(densearray)

learning_rate=[0.3]

if len(gpuarray)==0:
	logging.error('No GPUs specified')
	sys.exit()
if ndense==0:
	logging.error('No dense layer indices')
	sys.exit()
if ncolors==0:
	logging.error('No colors')
	sys.exit()
if options.fraction<0.0 or options.fraction>1.0:
	logging.error('Fraction out of range '+str(options.fraction))
	sys.exit()
if options.mode not in ['train','test']:
	logging.error('Invalid mode '+options.mode)
	sys.exit()
if options.signal=='None' or options.background=='None':
	logging.error('Need to specify both a signal and a background input file')
	sys.exit()

##------------------------------------------------------------------------------
#FUNCTIONS
##------------------------------------------------------------------------------

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model_file, 'rb') as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

def load_array(Array):
  print('Loading signal and background arrays ...')
  print('-----------'*10)
  data=np.load(image_array_dir_in+Array)
  print(type(data))
  return data

def expand_array(images):
  Nimages=len(images)
  expandedimages=np.zeros((Nimages,img_rows,img_cols,ncolors))
  for i in range(Nimages):
    npart = len(images[i])
    for j in range(npart):
       for nn in range(ncolors):
           expandedimages[i,images[i][j][0][0],images[i][j][0][1]][nn] = images[i][j][1][colarray[nn]]
  expandedimages=expandedimages.reshape(Nimages,img_rows,img_cols,ncolors)
  return expandedimages

def prepare_keras(xlist,ylist):
  yforkeras = keras.utils.to_categorical(ylist, num_classes)
  print('-----------'*10)
  print('Preparing inputs for keras...')
  xarray = np.array(xlist)
  yarray = np.array(yforkeras)
  print(xarray.shape, 'train sample shape')
  print('-----------'*10)
  return xarray,yarray

class DataGenerator(object):
  print('Generates data for Keras')
  def __init__(self, dim_x = img_rows, dim_y = img_cols,  batch_size = my_batch_size, shuffle = False):
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, N_train):
    while True:
      print('Number of training images:',N_train)
      imax=int(N_train/self.batch_size)
      print('Number of minibatches =',imax)
      print('\n'+'-----------'*10)
      print('////////////'*10)

      traingenerator=(json.loads(s) for s in open(trainfilename))
      for i in range(imax):
          x_val=[]
          y_val=[]
          z_val=[]
          for ijet in range(self.batch_size):
             xy=next(traingenerator)
             x_val.append(xy[0])
             y_val.append(xy[1])
             z_val.append([])
             for iden in densearray:
               z_val[-1].append(xy[iden])
          y_val=keras.utils.to_categorical(y_val, num_classes)
          x_val=np.array(x_val)
          y_val=np.array(y_val)
          z_val=np.array(z_val)

          images=expand_array(x_val)
          yield [images,z_val], y_val

  def valgenerate(self, N_val):
    while True:
      print('Number of validation images:',N_val)
      imax=int(N_val/self.batch_size)
      print('\n'+'-----------'*10)
      print('////////////'*10)

      valgenerator=(json.loads(s) for s in open(valfilename))
      for i in range(imax):
          x_val=[]
          y_val=[]
          z_val=[]
          for ijet in range(self.batch_size):
             xy=next(valgenerator)
             x_val.append(xy[0])
             y_val.append(xy[1])
             z_val.append([])
             for iden in densearray:
               z_val[-1].append(xy[iden])

          y_val=keras.utils.to_categorical(y_val, num_classes)
          x_val=np.array(x_val)
          y_val=np.array(y_val)
          z_val=np.array(z_val)

          images=expand_array(x_val)
          yield [images,z_val], y_val

##------------------------------------------------------------------------------
# DEFINE THE MODEL ARCHITECTURE
##------------------------------------------------------------------------------

input_shape_c = Input(shape=(img_rows, img_cols, ncolors))
input_shape_btag = Input(shape=(ndense,))

devstr = '/cpu:0'
if len(gpuarray)==1:
    devstr = '/gpu:0'

with tf.device(devstr):
  conv = Conv2D(128, kernel_size=(4,4),activation='relu',padding='same',name='Conv1')
  layers = conv(input_shape_c)
  layers = ZeroPadding2D(padding=(1, 1))(layers)
  layers = Conv2D(64, (4,4), activation='relu',name='Conv2')(layers)
  layers = MaxPooling2D(pool_size=(2, 2))(layers)
  layers = ZeroPadding2D(padding=(1, 1))(layers)
  layers = Conv2D(64, (4,4), activation='relu',name='Conv3')(layers)
  layers = ZeroPadding2D(padding=(1, 1))(layers)
  layers = Conv2D(64, (4,4), activation='relu',name='Conv4')(layers)
  layers = MaxPooling2D(pool_size=(2, 2))(layers)

  layers = Flatten()(layers)
  layers = Dense(64,activation='relu',name='Dense00')(layers)
  layersbtag = Dense(64,activation='relu',name='Dense10')(input_shape_btag)

  model_12 = keras.layers.concatenate([layers, layersbtag])
  model_12 = Dense(256, activation='relu',name='Dense20')(model_12)
  model_12 = Dense(256, activation='relu',name='Dense21')(model_12)
  model_12 = Dense(256, activation='relu',name='Dense22')(model_12)

  finalmodel = Dense(num_classes,  activation = 'softmax',name='Final')(model_12)

Adadelta=keras.optimizers.Adadelta(lr=learning_rate[0], rho=0.95, epsilon=1e-08, decay=0.0)
model = Model([input_shape_c, input_shape_btag], finalmodel)

#multi gpu support in testing -- need to do it this way in order to save weights
if len(gpuarray)>1:
	modeltr = multi_gpu_model(model, gpus=len(gpuarray))
	modeltr.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adadelta,
              metrics=['categorical_accuracy'])
	modeltr.summary()
else:
	modeltr = model
	modeltr.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adadelta,
              metrics=['categorical_accuracy'])
	modeltr.summary()

sd=[]
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self,epoch, logs={}):
    self.starttime=time.time()
  def on_epoch_end(self,epoch, logs={}):
    self.logs.append(time.time()-self.starttime)

class LossHistory(keras.callbacks.Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_train_begin(self, logs={}):
        self.loss = [1000000.] #Initial value of the val loss function
        self.acc = [1000000.] #Initial value of the val loss function
        self.val_loss = [1000000.] #Initial value of the val loss function
        self.val_acc = [1000000.] #Initial value of the val loss function

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss')) # We append the val loss of the last epoch to losses
        self.acc.append(logs.get('acc')) # We append the val loss of the last epoch to losses
        self.val_loss.append(logs.get('val_loss')) # We append the val loss of the last epoch to losses
        self.val_acc.append(logs.get('val_acc')) # We append the val loss of the last epoch to losses

def step_decay(losses):
    if len(history.val_loss)>=2 and float(np.array(history.val_loss[-2])-np.array(history.val_loss[-1]))<0.0005:
        lrate=learning_rate[-1]/np.sqrt(2)
        learning_rate.append(lrate)
    else:
        lrate=learning_rate[-1]

    if len(history.val_loss)>=2:
      print('\n loss[-2] = ',np.array(history.val_loss[-2]))
      print('\n loss[-1] = ',np.array(history.val_loss[-1]))
      print('\n loss[-2] - loss[-1] = ',float(np.array(history.val_loss[-2])-np.array(history.val_loss[-1])))

    print('\n Learning rate =',lrate)
    print('------------'*10)

    return lrate
history=LossHistory(modeltr)
lrate=keras.callbacks.LearningRateScheduler(step_decay)
# Get new learning rate
early_stop=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=5, verbose=0, mode='auto')
# patience -- means that if there is no improvement in the cross-validation accuracy greater that min_delta within the following 3 epochs, then it stops
checkpoint=keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb = TimingCallback()


##------------------------------------------------------------------------------
# TRAIN THE MODEL OR LOAD TRAINED WEIGHTS
##------------------------------------------------------------------------------

weights_dir = 'weights/'
os.system('mkdir -p '+weights_dir)



print('Getting the length of the signal and background files')
Nsignal=0
for s in open(image_array_dir_in+signalfilename):
  Nsignal=Nsignal+1
Nbackground=0
for s in open(image_array_dir_in+backgroundfilename):
  Nbackground=Nbackground+1

print('total number of signal jets:',Nsignal)
print('total number of background jets:',Nbackground)
Njets=min([Nsignal,Nbackground])
print('Njets',Njets)

train_frac_rel=0.6
val_frac_rel=0.2
test_frac_rel=0.2

train_frac=train_frac_rel
val_frac=train_frac+val_frac_rel
test_frac =val_frac+test_frac_rel

Ntrain=int(train_frac_rel*Njets*sample_relative_size)
Nval=int(val_frac_rel*Njets*sample_relative_size)
Ntest=int(test_frac_rel*Njets*sample_relative_size)

print('Size of training set:',2*Ntrain)
print('Size of validation set:',2*Nval)
print('Size of test set:',2*Ntest)

trainfilename='train_sample_'+str(Ntrain)+'_'+str(Nval)+'_'+str(Ntest)+'.dat'
valfilename='validation_sample_'+str(Ntrain)+'_'+str(Nval)+'_'+str(Ntest)+'.dat'
testfilename='test_sample_'+str(Ntrain)+'_'+str(Nval)+'_'+str(Ntest)+'.dat'

savename = 'epochs_'+str(epochs)+'_Ntrain_'+str(Ntrain)+'_'+name_sg.replace('.dat','_')+name_bg.replace('.dat','_')+post

if not options.skipgen:
  shufobj  = shf.pyshuffle(image_array_dir_in+signalfilename,image_array_dir_in+backgroundfilename,str(Ntrain),str(Nval),str(Ntest))
  shufobj.run()
  print('------------'*10)
  print('running shuffle')
  print('------------'*10)
  os.system('/localhome/knash/terashuf/terashuf < ' + trainfilename + ' > shuffled_'+trainfilename)
  os.system('mv ' + 'shuffled_'+trainfilename+' '+ trainfilename)

Ntrain=2*Ntrain
Nval=2*Nval
Ntest=2*Ntest

if options.load!='None':
  my_weights=options.load
  WEIGHTS_FNAME=my_weights
  if os.path.exists(WEIGHTS_FNAME):
      print('------------'*10)
      print('Loading existing weights',WEIGHTS_FNAME)
      print('------------'*10)
      modeltr.load_weights(WEIGHTS_FNAME)
      print('done')
  else:
    print('Weight file not found')

##------------------------------------------------------------------------------
# TRAIN THE MODEL
##------------------------------------------------------------------------------

if mode=='train':

  saveweightname=weights_dir+'cnn_weights_'+savename+'.hdf'

  train_x_train_y = DataGenerator().generate(Ntrain)
  val_x_val_y = DataGenerator().valgenerate(Nval)

  my_steps_per_epoch= int(Ntrain/my_batch_size)
  print('my_steps_per_epoch =',my_steps_per_epoch)
  valsteps=int(Nval/my_batch_size)
  print(valsteps)

  modeltr.fit_generator(generator = train_x_train_y,
                    steps_per_epoch = my_steps_per_epoch, #This is the number of files that we use to train in each epoch
                    epochs=epochs,
                    verbose=1,
                    validation_data =val_x_val_y,
                    validation_steps = valsteps,
                    callbacks=[history,lrate,early_stop,checkpoint,cb])#,
  print(cb.logs)
  print('------------'*10)
  print('Weights filename =',saveweightname)
  print('------------'*10)
if len(gpuarray)>1:
	model.save_weights(saveweightname, overwrite=True)
else:
	modeltr.save_weights(saveweightname, overwrite=True)

print('------------'*10)

##------------------------------------------------------------------------------
# ANALIZE RESULTS
##------------------------------------------------------------------------------

#LOAD LIBRARIES
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import pandas as pd
print('Computing test accuracy and ROC curve...')

##------------------------------------------------------------------------------
# PLOT DATA
##------------------------------------------------------------------------------

ROC_plots_dir = 'analysis/ROC/'
os.system('mkdir -p '+ROC_plots_dir)

##------------------------------------------------------------------------------
# PREDICT OUTPUT PROBABILITIES
##------------------------------------------------------------------------------

tempgenerator=(json.loads(s) for s in open(testfilename))
tempbatchsize=min([10000,Ntest])
nbatches=int(Ntest/tempbatchsize)
Y_Pred_prob= np.empty((0, 2))
y_test=np.empty((0, 2))
Ntesttry=0
for ibatch in range(nbatches+1):
  x_test_batch=[]
  y_test_batch=[]
  z_test_batch=[]
  ijetmax=min([tempbatchsize,Ntest-Ntesttry])
  if(ijetmax>0):
    for ijet in range(ijetmax):
      Ntesttry+=1
      xy=next(tempgenerator)
      x_test_batch.append(xy[0])
      y_test_batch.append(xy[1])
      z_test_batch.append([])
      for iden in densearray:
        z_test_batch[-1].append(xy[iden])
    x_test_batch,y_test_batch= prepare_keras(x_test_batch,y_test_batch)
    testimages=expand_array(x_test_batch)
    z_test_batch=np.array(z_test_batch)
    Y_Pred_batch = model.predict([testimages,z_test_batch])
    Y_Pred_prob=np.concatenate((Y_Pred_prob,Y_Pred_batch))
    y_test=np.concatenate((y_test,y_test_batch))

print('begin printing CNN output')
print('------------'*10)
ypredfile=open('ypred.dat','w')
print('------------'*10)

difflist=[(1-int(x[0][0]/0.5))-x[1][0] for x in zip(Y_Pred_prob,y_test)]
print('Test accuracy = ',float(np.count_nonzero(np.array(difflist)))/float(Ntesttry) )

# Predict output probability for each class (signal or background) for the image
y_Pred = np.argmax(Y_Pred_prob, axis=1)
y_Test = np.argmax(y_test, axis=1)
print('Predicted output from the CNN (0 is signal and 1 is background) = \n',y_Pred[0:15])
print('y_Test (True value) =\n ',y_Test[0:15])
print('y_Test lenght', len(y_Test))
print('------------'*10)

#Print classification report
print(classification_report(y_Test, y_Pred))
print('------------'*10)

# Calculate a single probability of tagging the image as signal
out_prob=[]
for i_prob in range(len(Y_Pred_prob)):
    out_prob.append((Y_Pred_prob[i_prob][0]-Y_Pred_prob[i_prob][1]+1)/2)

print('Predicted probability of each output neuron = \n',Y_Pred_prob[0:15])
print('------------'*10)
print('Output of tagging image as signal = \n',np.array(out_prob)[0:15])
print('------------'*10)

np.savetxt('outprob.csv', np.array(out_prob), delimiter=',')
np.savetxt('inprob.csv', np.array(y_test), delimiter=',')

# Make ROC with area under the curve plot
def generate_results(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score,pos_label=0, drop_intermediate=False)
    print('Thresholds[0:6] = \n',thresholds[:6])
    print('Thresholds lenght = \n',len(thresholds))
    print('fpr lenght',len(fpr))
    print('tpr lenght',len(tpr))
    rocnums=list(zip(fpr,tpr))
    rocout=open(ROC_plots_dir+'roc_'+savename+'.csv','wb')
    np.savetxt(rocout,rocnums,fmt='%10.5g',delimiter=',')

    print('------------'*10)
    roc_auc = auc(fpr, tpr)

    print('AUC =', np.float128(roc_auc))
    print('------------'*10)

generate_results(y_Test, out_prob)
print('FINISHED.')
if len(gpuarray)>1:
	model.save(extrastring+'.h5')
else:
	modeltr.save(extrastring+'.h5')

print('-----------'*10)
print('Code execution time = %s minutes' % ((time.time() - start_time)/60))
print('-----------'*10)
