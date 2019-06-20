import ROOT
from ROOT import TH1F,TH2F,TGraph,TCanvas,TLatex,TMarker
import glob
import copy
import random
import sys
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


datfiles = glob.glob(str(sys.argv[1]))
datfilessig = glob.glob(str(sys.argv[2]))

Masshistpre =  TH1F("Masshistpre",	"Masshistpre",		100, 0,3*172.5 )
MasshistpostL =  TH1F("MasshistpostL",	"MasshistpostL",		100, 0,3*172.5 )
MasshistpostM =  TH1F("MasshistpostM",	"MasshistpostM",		100, 0,3*172.5 )
MasshistpostT =  TH1F("MasshistpostT",	"MasshistpostT",		100, 0,3*172.5 )
Masshist2d =  TH2F("Masshist2d",	"Masshist2d",		40, 0,3,200,0,1 )
Masshist2dptpre =  TH2F("Masshist2dptpre",	"Masshist2dptpre",		50, 0,3,20,0,2 )
Masshist2dptpostL =  TH2F("Masshist2dptpostL",	"Masshist2dptpostL",		50, 0,3,20,0,2 )
Masshist2dptpostM =  TH2F("Masshist2dptpostM",	"Masshist2dptpostM",		50, 0,3,20,0,2 )
Masshist2dptpostT =  TH2F("Masshist2dptpostT",	"Masshist2dptpostT",		50, 0,3,20,0,2 )
dnnsig =  TH1F("dnnsig",	"dnnsig",		200, 0,1 )
dnnbkg =  TH1F("dnnbkg",	"dnnbkg",		200, 0,1 )

qcdN = []
qcdNwmass = []
qcdD = []
sigN = []
sigNwmass = []
sigD = []
reso=700
for iitop in xrange(reso):
	qcdD.append(0.)
	qcdN.append(0.)
	qcdNwmass.append(0.)
	sigNwmass.append(0.)
	sigN.append(0.)
	sigD.append(0.)



wmeff = 0.0
wmeffD = 0.0
linemax = 200000000
ptlims=[0.5,0.75]

#t
#masslims = [0.65,1.3]
#H
#masslims = [0.5,0.9]
#W
masslims = [55./172.5,95./172.5]

for datfile in datfilessig:
	with open(datfile, "r") as ins:
		nline =0

 		for line in ins:
			#print nline
 			if nline >linemax:break


			curpt = float(line.split(",")[-1])
			if not ((curpt>ptlims[0]) and (curpt<ptlims[1])):
				continue
			curmass = float(line.split(",")[-2])
			#print line
			curnn = float(line.split(",")[-3])
			dnnsig.Fill(curnn)
			wmeffD+=1.0
			if (curmass>masslims[0]) and (curmass<masslims[1]):
					wmeff+=1.0
					
			#print curmass,curnn
			for iitop in xrange(reso):
				sigD[iitop]+=1.0
				nnval = 1.0-(float(iitop)/float(reso))**5


				if curnn>nnval:
						sigN[iitop]+=1.0

				if curnn>nnval and (curmass>masslims[0]) and (curmass<masslims[1]):
						sigNwmass[iitop]+=1.0
						nline+=1

print "mass cut eff",float(wmeff)/float(wmeffD)
npre=0
npost=0
wbgmeff = 0.0
wbgmeffD = 0.0
for datfile in datfiles:
	with open(datfile, "r") as ins:
		nline =0
 		for line in ins:
			#print nline
 			if nline >linemax:break

			curpt = float(line.split(",")[-1])
			if not ((curpt>ptlims[0]) and (curpt<ptlims[1])):
				continue
			curmass = float(line.split(",")[-2])
			curnn = float(line.split(",")[-3])
			dnnbkg.Fill(curnn)
			wbgmeffD+=1.0
			if (curmass>masslims[0]) and (curmass<masslims[1]):
					wbgmeff+=1.0
			for iitop in xrange(reso):
				qcdD[iitop]+=1.0
				nnval = 1.0-(float(iitop)/float(reso))**5
				
				if curnn>nnval:
						qcdN[iitop]+=1.0
				if curnn>nnval and (curmass>masslims[0]) and (curmass<masslims[1]):
						qcdNwmass[iitop]+=1.0
						nline+=1
		
			Masshistpre.Fill(curmass*172.5)
			Masshist2d.Fill(curmass,curnn)
			Masshist2dptpre.Fill(curmass,curpt)
			if float(curnn)>0.6:
				MasshistpostL.Fill(curmass*172.5)
				Masshist2dptpostL.Fill(curmass,curpt)
			if float(curnn)>0.9:
				MasshistpostM.Fill(curmass*172.5)
				#print line
				Masshist2dptpostM.Fill(curmass,curpt)
			if float(curnn)>0.99:
				MasshistpostT.Fill(curmass*172.5)
				Masshist2dptpostT.Fill(curmass,curpt)

bgeff=float(wbgmeff)/float(wbgmeffD)

print "mass cut eff",bgeff
print "donelooping"
ROC=TGraph()
ROCwmass=TGraph()
for iitop in xrange(reso):

	ROC.SetPoint(iitop,sigN[iitop]/sigD[iitop],qcdN[iitop]/qcdD[iitop])
	ROCwmass.SetPoint(iitop,sigNwmass[iitop]/sigD[iitop],qcdNwmass[iitop]/qcdD[iitop])

print "nncut bkg red"
print "L",bgeff*100.*MasshistpostL.Integral()/(Masshistpre.Integral()+MasshistpostL.Integral()),"%"
print "M",bgeff*100.*MasshistpostM.Integral()/(Masshistpre.Integral()+MasshistpostM.Integral()),"%"
print "T",bgeff*100.*MasshistpostT.Integral()/(Masshistpre.Integral()+MasshistpostT.Integral()),"%"
output = ROOT.TFile("histosQCDmassana_"+str(sys.argv[1]).replace(".dat","")+".root","recreate")
output.cd()
prelim = TLatex()
prelim.SetNDC()



Masshistpre.Write("Masshistpre")
MasshistpostL.Write("MasshistpostL")
MasshistpostM.Write("MasshistpostM")
MasshistpostT.Write("MasshistpostT")
Masshist2dptpostL.Write("Masshist2dptpostL")
Masshist2dptpostM.Write("Masshist2dptpostM")
Masshist2dptpostT.Write("Masshist2dptpostT")
c1 = TCanvas("c1")
Masshistpre.Scale(1.0/Masshistpre.Integral())
MasshistpostM.Scale(1.0/MasshistpostM.Integral())

Masshistpre.SetLineColor(1)
MasshistpostM.SetLineColor(2)
Masshistpre.SetTitle(";M_{SD} GeV;A.U.")
Masshistpre.SetStats(0)
Masshistpre.Draw("hist")
MasshistpostM.Draw("samehist")
prelim.DrawLatex( 0.3, 0.75, "500 GeV < p_{T} < 2000 GeV" )
c1.Write("masscomp")

Masshist2d.Write("Masshist2d")

ROC.SetMinimum(0.0001)
ROC.Write("ROC")
ROCwmass.Write("ROCwmass")
ROCwmass.SetMinimum(0.0001)
ROCwmass.SetMaximum(1.0)
ROCwmass.GetXaxis().SetRangeUser(0.0,1.0)
ROCwmass.SetLineColor(2)
ROCwmass.SetLineWidth(2)
ROCwmass.SetTitle(";signal efficiency;background rejection")
c2 = TCanvas("c2")




ROCwmass.Draw()
prelim.DrawLatex( 0.15, 0.85, "500 GeV < p_{T} < 2000 GeV" )

prelim.DrawLatex( 0.44, 0.55, "N_{2}^{DDT}" )
TM = TMarker(0.4, 0.01, 23)
TM.SetMarkerSize(2)
TM.SetMarkerColor(4)
TM.Draw()
c2.SetLogy()
c2.Write("rocwmass")
ROC.Draw()

c2.Write("roccomp")

output.Write()
output.Close()
