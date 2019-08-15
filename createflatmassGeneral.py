import ROOT
from ROOT import TH1F,TH2F
import glob
import copy
import random
from optparse import OptionParser
import json
import subprocess
import sys
import FWCore.ParameterSet.VarParsing as opts
options = opts.VarParsing ('analysis')
options.register('sigf',
     'None',
     opts.VarParsing.multiplicity.singleton,
     opts.VarParsing.varType.string,
     'sigfname')
options.register('bkgf',
     'None',
     opts.VarParsing.multiplicity.singleton,
     opts.VarParsing.varType.string,
     'bkgfname')
options.register('cust',
     'top',
     opts.VarParsing.multiplicity.singleton,
     opts.VarParsing.varType.string,
     'cust')
options.register('maxev',
     10000000,
     opts.VarParsing.multiplicity.singleton,
     opts.VarParsing.varType.int,
     'cust')
options.register('skipcut',
     False,
     opts.VarParsing.multiplicity.singleton,
     opts.VarParsing.varType.bool,
     'skipcut')
options.register('ishotvr',
     False,
     opts.VarParsing.multiplicity.singleton,
     opts.VarParsing.varType.bool,
     'ishotvr')
options.parseArguments()
cust= options.cust
ishotvr=options.ishotvr
nping=1500000
bkgstr = (options.bkgf).replace(".dat","").replace(",","")
sigstr = (options.sigf).replace(".dat","").replace(",","")


truthmod = [0]
tblack = []
limmass = [0.0,4.0]


if cust =="multi":
	#W2 Z3 H2 t3
	truthmod = [0,2,5,7]
	limpt = [0.1,1.8]

if cust =="vfull":
	#W2 Z3 H2 
	truthmod = [0,2,5]
	limpt = [0.1,1.8]

if cust in ["w","z"]:
	limpt = [0.1,1.8]
	

if cust in ["ww","wwlep"]:
	limpt = [0.1,1.8]
	if cust =="ww":
		tblack = [6,7]
	if cust =="wwlep":
		truthmod = [-2]
		tblack = [4,5]

	
if cust in ["pho"]:
	limpt = [0.15,1.8]
	
if cust in ["h","hbb","hww","hwwlep","hfull"]:
	limpt = [0.15,1.8]
	
	if cust =="hbb":
		truthmod = [-1]
	if cust =="hww":
		truthmod = [-2]
		tblack = [8,9]
	if cust =="hwwlep":
		truthmod = [-4]
		tblack = [6,7]
if cust=="top":
	limpt = [0.15,1.8]


if options.ishotvr:
	print "HotVR"
	limpt = [0.1,1.8]
	truthmod[0] = truthmod[0]-3

maxev=options.maxev
sigfiles=(options.sigf).split(",")
datfilestt=[]
for ss in sigfiles:
	datfilestt += glob.glob(ss)
Masshisttt =  TH1F("Masshisttt",	"Masshisttt",		200, 0,4 )
pthisttt =  TH1F("pthisttt",	"pthisttt",		200, 0.0,3.0 )
mpt2dtt =  TH2F("mpt2dtt",	"mpt2dtt",		200, limmass[0],limmass[1],25, limpt[0],limpt[1] )
mpt2dtt.Sumw2()
print "counting sigs"
nevs = [maxev]
for datfile in datfilestt:
	with open(datfile, "r") as ins:
        	for i, l in enumerate(ins):
			#print i
			if i>maxev:break
            		pass
	print i + 1
	nevs.append(i + 1)
neventsttbar = min(nevs)
print "nev",neventsttbar
neventsbkg = neventsttbar*len(datfilestt)

print "nevsig",neventsttbar
print "SIG"
ifile=0
for datfile in datfilestt:
	with open(datfile, "r") as ins:
		nline =0
		nlinept =0
		totev = 0 
 		for line in ins:

			#line = '[[[['+line.split('[[[[')[-1]
			try:
				jsonl = json.loads(line)
			except:
				continue

			if nline==0:nlineinit=len(jsonl)

			if len(jsonl)!=nlineinit:
				print "BL" 
				break

			curmass=jsonl[-3]
			curpt=jsonl[-2]

			if jsonl[-4] in tblack:
				continue
			#print jsonl
			#print jsonl[-4],truthmod[ifile]
			jsonl[-4]=jsonl[-4]+truthmod[ifile]
			#print jsonl
			#print
			if not (limpt[0]<curpt<limpt[1]):
				continue
		
			jsonl[-4]=jsonl[-4]+truthmod[ifile]
			if not (limpt[0]<curpt<limpt[1]):
				continue
		
			
			pthisttt.Fill(curpt)
			if not (limmass[0]<curmass<limmass[1]):
				continue
			if nline>nping:
				break
			if nline%50000==0:
				print nline,"lines"

			mpt2dtt.Fill(curmass,curpt)

			Masshisttt.Fill(curmass)
			nline+=1
	ifile+=1


bkgfiles=(options.bkgf).split(",")
datfilesQCD=[]
for bb in bkgfiles:
	datfilesQCD += glob.glob(bb)

MasshistQCD =  TH1F("MasshistQCD",	"MasshistQCD",		200, 0,4 )
pthistQCD =  TH1F("pthistQCD",	"pthistQCD",		200, 0.0,3.0 )
mpt2dQCD =  TH2F("mpt2dQCD",	"mpt2dQCD",	200,limmass[0],limmass[1],25, limpt[0],limpt[1] )
mpt2dQCD.Sumw2()
print "QCD"
for datfile in datfilesQCD:
	with open(datfile, "r") as ins:
		
		nline =0
 		for line in ins:
			
			try:
				jsonl = json.loads(line)
			except:
				continue

			if len(jsonl)!=nlineinit:
				print "BL" 
				break


			curmass=jsonl[-3]
			curpt=jsonl[-2]


			if not (limpt[0]<curpt<limpt[1]):
				continue


			pthistQCD.Fill(curpt)
			if not (limmass[0]<curmass<limmass[1]):
				continue




			if nline>nping*len(datfilestt):
				break
			if nline%50000==0:
				print nline,"lines"
			
			mpt2dQCD.Fill(curmass,curpt)
			MasshistQCD.Fill(curmass)
			nline+=1


rathistN = copy.deepcopy(Masshisttt)
rathistN.Divide(MasshistQCD)
rathistN.Scale(1.0/rathistN.GetMaximum())

#mpt2dQCD.Smooth()

rathist2dN = copy.deepcopy(mpt2dtt)
rathist2dN.Divide(mpt2dQCD)
rathist2dN.Scale(1.0/rathist2dN.GetMaximum())

rathistptN = copy.deepcopy(pthisttt)
rathistptN.Divide(pthistQCD)
rathistptN.Scale(1.0/rathistptN.GetMaximum())



MDMasshistsigpost =  TH1F("MDMasshistsigpost",	"MDMasshistsigpost",		200, 0,4 )
MDpthistsigpost =  TH1F("MDpthistsigpost",	"MDpthistsigpost",		200, 0.0,3.0 )

fwot = open('orthtest'+cust+'constmass.dat', 'w')
fwotpt = open('orthtest'+cust+'constpt.dat', 'w')

fws = open(sigstr+cust+'constmass.dat', 'w')
fwspt = open(sigstr+cust+'constpt.dat', 'w')
print "nevsig",neventsttbar
print "SIG"
ifile=0
nallline=0
for datfile in datfilestt:
	with open(datfile, "r") as ins:
		nline =0
		nlinept =0
 		for line in ins:

			nallline+=1
			if nallline>neventsttbar:
				break
			writetotest=False
			if nallline%5==0:
				writetotest=True
			if nline>maxev:break
			
			
			try:
				jsonl = json.loads(line)
			except:
				continue
			
			if nline==0:nlineinit=len(jsonl)

			if len(jsonl)!=nlineinit:
				print "BL" 
				break
			curmass=jsonl[-3]
			curpt=jsonl[-2]

			if jsonl[-4] in tblack:
				continue
			jsonl[-4]=jsonl[-4]+truthmod[ifile]
				
			
			if not (limpt[0]<curpt<limpt[1]):
				continue
			if writetotest:
				fwotpt.write(json.dumps(jsonl) + "\n")
			else:
				fwspt.write(json.dumps(jsonl) + "\n")
			
			nlinept+=1

			if not (limmass[0]<curmass<limmass[1]):
				continue
			
			if nline%20000==0:
				print nline,"lines"
					
			MDMasshistsigpost.Fill(curmass)
			MDpthistsigpost.Fill(curpt)
			if writetotest:
				fwot.write(json.dumps(jsonl) + "\n")
			else:
				fws.write(json.dumps(jsonl) + "\n")

			nline+=1
	ifile+=1




fws.close()
fwspt.close()


fw = open(bkgstr+cust+'constmass.dat', 'w')
fwpt = open(bkgstr+cust+'constpt.dat', 'w')
print "Run the QCD cut"
nlinepttowrite = nlinept

MasshistQCDpost =  TH1F("MasshistQCDpost",	"MasshistQCDpost",		200, 0,4 )
pthistQCDpost =  TH1F("pthistQCDpost",	"pthistQCDpost",		200, 0.0,3.0 )

MDMasshistQCDpost =  TH1F("MDMasshistQCDpost",	"MDMasshistQCDpost",		200, 0,4 )
MDpthistQCDpost =  TH1F("MDpthistQCDpost",	"MDpthistQCDpost",		200, 0.0,3.0 )
nallline=0
for datfile in datfilesQCD:
	if options.skipcut:
		break
	with open(datfile, "r") as ins:
		nline =0
		nlinept =0
		nlineprecut =-1


 		for line in ins:
			
			nallline+=1
			writetotest=False
			if nallline%5==0:
				writetotest=True
			try:
				jsonl = json.loads(line)
			except:
				continue

			nlineprecut+=1
			if nlineprecut%200000==0:
				print nline,"lines"
			if len(jsonl)!=nlineinit:
				print "BL" 
				break
			curmass=jsonl[-3]
			curpt=jsonl[-2]
			pyrand = random.random()



			probpt = rathistptN.GetBinContent(rathistptN.FindBin(curpt))

			if not (limpt[0]<curpt<limpt[1]):
				continue
			if nlinept<nlinepttowrite:
				if pyrand<(probpt):
				
					MasshistQCDpost.Fill(curmass)
					pthistQCDpost.Fill(curpt)
				
					if writetotest:
						fwotpt.write(json.dumps(jsonl) + "\n")
					else:
						fwpt.write(json.dumps(jsonl) + "\n")
					nlinept+=1		
		
			if not (limmass[0]<curmass<limmass[1]):
				continue
			Ybin=rathist2dN.GetYaxis().FindBin(curpt)
			Xbin=rathist2dN.GetXaxis().FindBin(curmass)
			
			probm=rathist2dN.Interpolate(curmass,curpt)
			if pyrand<(probm):
				 
				if nline<neventsbkg:
					if writetotest:
						fwot.write(json.dumps(jsonl) + "\n")
					else:
						fw.write(json.dumps(jsonl) + "\n")
				
					MDMasshistQCDpost.Fill(curmass)
					MDpthistQCDpost.Fill(curpt)
					nline+=1
				else:
					break	
fw.close()
fwpt.close()
if len(datfilestt)>1:
	commands=[]
	commands.append("shuf "+sigstr+"constmass.dat"+" -o "+(cust+"constmass_shuf.dat"))
	commands.append("shuf "+sigstr+"constpt.dat"+" -o "+(cust+"constpt_shuf.dat"))
	commands.append("rm "+sigstr+"constmass.dat")
	commands.append("rm "+sigstr+"constpt.dat")
	for s in commands :
    		print 'executing ' + s
   	 	subprocess.call( [s], shell=True )



output = ROOT.TFile(bkgstr+cust+"histosQCDflatmass.root","recreate")
output.cd()
Masshisttt.Write("Masshisttt")
MasshistQCD.Write("MasshistQCD")
MasshistQCDpost.Write("MasshistQCDpost")
rathistN.Write("rathistN")
rathist2dN.Write("rathist2dN")
pthisttt.Write("pthisttt")
pthistQCD.Write("pthistQCD")
pthistQCDpost.Write("pthistQCDpost")
rathistptN.Write("rathistptN")

MDMasshistQCDpost.Write("MDMasshistQCDpost")
MDpthistQCDpost.Write("MDpthistQCDpost")

output.Write()
output.Close()
