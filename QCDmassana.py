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
ptbinsform = [[0.15,0.25],[0.25,0.5],[0.5,0.75],[0.75,1.0],[1.0,1.25]]
Mplots={}
for ptb in ptbinsform:
	ptstrs = [str(int(2000.0*ptb[0])),str(int(2000.0*ptb[1]))]
	ptstr = ptstrs[0]+"to"+ptstrs[1]
	Mplots[ptstr+"pre"] =  TH1F("Masshistpre"+ptstr,	"Masshistpre"+ptstr,		100, 0,3*172.5 )
	Mplots[ptstr+"postL"] =  TH1F("MasshistpostL"+ptstr,	"MasshistpostL"+ptstr,		100, 0,3*172.5 )
	Mplots[ptstr+"postM"] =  TH1F("MasshistpostM"+ptstr,	"MasshistpostM"+ptstr,		100, 0,3*172.5 )
	Mplots[ptstr+"postT"] =  TH1F("MasshistpostT"+ptstr,	"MasshistpostT"+ptstr,		100, 0,3*172.5 )
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
reso=600
for iitop in xrange(reso):
	qcdD.append(0.)
	qcdN.append(0.)
	qcdNwmass.append(0.)
	sigNwmass.append(0.)
	sigN.append(0.)
	sigD.append(0.)



wmeff = 0.0
wmeffD = 0.0
linemax = 2000000
ptlims=[0.5,0.75]
Drawmarker=False
cust="ww"
if cust=="top":
	#masslims = [0.63,1.28]
	masslims = [0.66,1.4]
	Drawmarker=False
if cust=="h":
	masslims = [0.5,0.9]
if cust=="w":
	masslims = [55./172.5,95./172.5]
	Drawmarker=True
if cust=="wwlep" or cust=="hwwlep" or cust=="hww" or cust=="ww":
	masslims = [0.3,99.0]

for datfile in datfilessig:
	with open(datfile, "r") as ins:
		nline =0

 		for line in ins:
			#print nline
 			if nline >linemax:break

			splitv = line.split(",")
			curpt = float(splitv[-1])
			if not ((curpt>ptlims[0]) and (curpt<ptlims[1])):
				continue
			curmass = float(splitv[-2])
			#print line
			curnn = float(splitv[-3])
			dnnsig.Fill(curnn)
			wmeffD+=1.0
			if (curmass>masslims[0]) and (curmass<masslims[1]):
					wmeff+=1.0
					
			#print curmass,curnn
			for iitop in xrange(reso):
				sigD[iitop]+=1.0
				nnval = 1.0-(float(iitop)/float(reso))**6


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
			splitv = line.split(",")
			curpt = float(splitv[-1])
			curmass = float(splitv[-2])
			curnn = float(splitv[-3])


			for ptb in ptbinsform:
				if ((curpt>ptb[0]) and (curpt<ptb[1])):
					ptstrs = [str(int(2000.0*ptb[0])),str(int(2000.0*ptb[1]))]
					ptstr = ptstrs[0]+"to"+ptstrs[1]
					Mplots[ptstr+"pre"].Fill(curmass*172.5)
					
					if float(curnn)>0.6:
						Mplots[ptstr+"postL"].Fill(curmass*172.5)
					
					if float(curnn)>0.9:
						Mplots[ptstr+"postM"].Fill(curmass*172.5)
						
					if float(curnn)>0.99:
						Mplots[ptstr+"postT"].Fill(curmass*172.5)
			if not ((curpt>ptlims[0]) and (curpt<ptlims[1])):
				continue
			Masshist2d.Fill(curmass,curnn)
			Masshist2dptpre.Fill(curmass,curpt)
			if float(curnn)>0.6:
				
				Masshist2dptpostL.Fill(curmass,curpt)
			if float(curnn)>0.9:
				
				Masshist2dptpostM.Fill(curmass,curpt)
			if float(curnn)>0.99:
				
				Masshist2dptpostT.Fill(curmass,curpt)


			dnnbkg.Fill(curnn)
			wbgmeffD+=1.0
			if (curmass>masslims[0]) and (curmass<masslims[1]):
					wbgmeff+=1.0
			for iitop in xrange(reso):
				qcdD[iitop]+=1.0
				nnval = 1.0-(float(iitop)/float(reso))**6
				
				if curnn>nnval:
						qcdN[iitop]+=1.0
				if curnn>nnval and (curmass>masslims[0]) and (curmass<masslims[1]):
						qcdNwmass[iitop]+=1.0
						nline+=1
		


bgeff=float(wbgmeff)/float(wbgmeffD)

print "mass cut eff",bgeff
print "donelooping"
ROC=TGraph()
ROCwmass=TGraph()
for iitop in xrange(reso):

	ROC.SetPoint(iitop,sigN[iitop]/sigD[iitop],qcdN[iitop]/qcdD[iitop])
	ROCwmass.SetPoint(iitop,sigNwmass[iitop]/sigD[iitop],qcdNwmass[iitop]/qcdD[iitop])
output = ROOT.TFile("histosQCDmassana_"+str(sys.argv[1]).replace(".dat","")+".root","recreate")
output.cd()
print "nncut bkg red"
prelim = TLatex()
prelim.SetNDC()
canvs = []
for ptb in ptbinsform:
	ptstrs = [str(int(2000.0*ptb[0])),str(int(2000.0*ptb[1]))]
	ptstr = ptstrs[0]+"to"+ptstrs[1]
	print ptstr,"L",bgeff*100.*Mplots[ptstr+"postL"].Integral()/(Mplots[ptstr+"pre"].Integral()+Mplots[ptstr+"postL"].Integral()),"%"
	print ptstr,"M",bgeff*100.*Mplots[ptstr+"postM"].Integral()/(Mplots[ptstr+"pre"].Integral()+Mplots[ptstr+"postM"].Integral()),"%"
	print ptstr,"T",bgeff*100.*Mplots[ptstr+"postT"].Integral()/(Mplots[ptstr+"pre"].Integral()+Mplots[ptstr+"postT"].Integral()),"%"
	 
	canvs.append(TCanvas("c"+ptstr))
	if Mplots[ptstr+"postM"].Integral()>0:
		Mplots[ptstr+"pre"].Scale(1.0/Mplots[ptstr+"pre"].Integral())
		Mplots[ptstr+"postM"].Scale(1.0/Mplots[ptstr+"postM"].Integral())

		Mplots[ptstr+"pre"].SetLineColor(1)
		Mplots[ptstr+"postM"].SetLineColor(2)
		Mplots[ptstr+"pre"].SetTitle(";M_{SD} GeV;A.U.")
		Mplots[ptstr+"pre"].SetStats(0)
		Mplots[ptstr+"pre"].Draw("hist")
		Mplots[ptstr+"postM"].Draw("samehist")
		prelim.DrawLatex( 0.3, 0.75, ptstrs[0]+" GeV < p_{T} < "+ptstrs[1]+" GeV" )
		canvs[-1].Write("masscomp"+ptstr)







Masshist2dptpostL.Write("Masshist2dptpostL")
Masshist2dptpostM.Write("Masshist2dptpostM")
Masshist2dptpostT.Write("Masshist2dptpostT")


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



ptlimstrs = [str(int(2000.0*ptlims[0])),str(int(2000.0*ptlims[1]))]
ROCwmass.Draw()
prelim.DrawLatex( 0.15, 0.85, ptlimstrs[0]+" GeV < p_{T} < "+ptlimstrs[1]+" GeV" )
if (cust=="w" or cust=="top") and Drawmarker:
	if cust=="w":
		prelim.DrawLatex( 0.44, 0.55, "N_{2}^{DDT}+msd" )
		TM = TMarker(0.4, 0.01, 23)
	if cust=="top":
		prelim.DrawLatex( 0.44, 0.60, "#tau_{32}+msd+subjetb" )
		TM = TMarker(0.4, 0.017, 23)
	TM.SetMarkerSize(2)
	TM.SetMarkerColor(4)
	TM.Draw()
c2.SetLogy()
c2.Write("rocwmass")
ROC.Draw()

c2.Write("roccomp")

dnnsig.Write()
dnnbkg.Write()
output.Write()
output.Close()
