class pyshuffle():
  def __init__(self,sigfile,bgfile,Ntrain,Nval,Ntest):
    self.sigfile = sigfile
    self.bgfile = bgfile
    self.Ntrain = Ntrain
    self.Nval = Nval
    self.Ntest = Ntest
    itest=Ntest+Nval
    itrain=Ntest+Nval+Ntrain
  def run(self):
    valf  = open('validation_sample_'+str(self.Ntrain)+'_'+str(self.Nval)+'_'+str(self.Ntest)+'.dat', 'w')
    traf  = open('train_sample_'+str(self.Ntrain)+'_'+str(self.Nval)+'_'+str(self.Ntest)+'.dat', 'w')
    tesf  = open('test_sample_'+str(self.Ntrain)+'_'+str(self.Nval)+'_'+str(self.Ntest)+'.dat', 'w')
    nline = 0
    with open(self.sigfile) as infile:
      for line in infile:
        if(0<=nline<int(self.Nval)):
          valf.write(line)
        if(int(self.Nval)<=nline<int(self.Nval)+int(self.Ntest)):
          tesf.write(line)
        if(int(self.Nval)+int(self.Ntest)<=nline<int(self.Nval)+int(self.Ntest)+int(self.Ntrain)):
          traf.write(line)
        nline+=1
    nline = 0
    with open(self.bgfile) as infile:
      for line in infile:
        if(0<=nline<int(self.Nval)):
          valf.write(line)
        if(int(self.Nval)<=nline<int(self.Nval)+int(self.Ntest)):
          tesf.write(line)
        if(int(self.Nval)+int(self.Ntest)<=nline<int(self.Nval)+int(self.Ntest)+int(self.Ntrain)):
          traf.write(line)
        nline+=1

		
