import numpy as np
import matplotlib.pyplot as plt
import mcmc_funct as mcmc
import subprocess
from astropy.io import fits


# Physical constants:
yr = 31556926.0               #  s 
psc = 3.09e18                 #  cm 
msolar = 1.99e33              #  g 
m_proton = 1.67262158e-24     #  g 
psi_0 = 2.026                 #
clight = 3.e10                #  cm/s 
m_electron = 9.10938188e-28   #  g 
e_charge = 4.8032068e-10      #  esu 
#erg = 624.150974             #  GeV 
hcgs = 6.626068e-27           #  ergs s [Planck's Constant] 
hev = 4.1356668e-15           #  eV s [Planck's Constant] 
kb = 1.380658e-16             #  ergs / K [Boltzmann's constant] 
mec2 = 510.998910             #  electron rest mass energy in keV 
sigmat = 6.6524586e-25        #  Thomson cross-section in cm^2 
mjy = 1.e-26                  #  ergs / s / cm^2 / Hz  
hev = 4.135667662*10**(-15)   #  Planck's constant[eV.s]
herg = 6.626068*10**-27       #  Planck's constant [ergs.s]
erg = 6.424*10**(11)          #  Erg to eV[eV/erg]
kpc = 3.086e+21               #  kpc to cm


probs=[]
pars=[]



with open('prob.dat') as f1:
    strprob= f1.readlines()
    for prob in strprob:
        probs.append(float(prob))
with open('predictobs.dat') as f3:
    predictobs=f3.readlines()
maxprob=max(probs)
print(maxprob)
ndx=np.argmax(probs)
with open ('chain.dat') as f2:
    for i in f2.readlines():
        #print(i)
        pars.append([float(x) for x in i.split()])



print(pars[ndx])
print('\n')
print(predictobs[ndx])

with open ('bestfit.txt','w') as fit:
    fit.write('Likelihood: '+str(maxprob)+'\n')
    fit.write('Pars: '+str(pars[ndx])+'\n')
    fit.write('Predictobs: '+str(predictobs[ndx])+'\n')

# chain=[]
# steps=[]
# par1=[]
# par2=[]
# par3=[]
# par4=[]
# par5=[]
# par6=[]
# par7=[]
# par8=[]
# par9=[]
# par10=[]
# par11=[]
# par12=[]
# for i in range(len(pars)):
#   par1.append(pars[i][0])
#   par2.append(pars[i][1])
#   par3.append(pars[i][2])
#   par4.append(pars[i][3])
#   par5.append(pars[i][4])
#   par6.append(pars[i][5])
#   par7.append(pars[i][6])
#   par8.append(pars[i][7])
#   par9.append(pars[i][8])
#   par10.append(pars[i][9])
#   par11.append(pars[i][10])
#   par12.append(pars[i][11])

# for j in range(int(len(pars)/200)):
#   for i in range(200):
#       steps.append(j)



#print(len(steps))
#print(len(chain1))
#print(steps)

        
# fig1=plt.figure()

# plt.subplot(321)
# plt.plot(steps,par1,color='k',alpha=0.5)
# plt.ylabel('Log[Esn]')

# plt.subplot(322)
# plt.plot(steps,par2,color='k',alpha=0.5)
# plt.ylabel('Log[Mej]')

# plt.subplot(323)
# plt.plot(steps,par3,color='k',alpha=0.5)
# plt.ylabel('Log[nism]')

# plt.subplot(324)
# plt.plot(steps,par4,color='k',alpha=0.5)
# plt.ylabel('brakind')

# plt.subplot(325)
# plt.plot(steps,par5,color='k',alpha=0.5)
# plt.ylabel('Log[tau]')

# plt.subplot(326)
# plt.plot(steps,par6,color='k',alpha=0.5)
# plt.ylabel('Log[Etab]')

# fig1.tight_layout(h_pad=0.0)
# plt.savefig('space-covered1.jpg')

# fig2=plt.figure()

# plt.subplot(321)
# plt.plot(steps,par7,color='k',alpha=0.5)
# plt.ylabel('Log[Emin]')

# plt.subplot(322)
# plt.plot(steps,par8,color='k',alpha=0.5)
# plt.ylabel('Log[Emax]')

# plt.subplot(323)
# plt.plot(steps,par9,color='k',alpha=0.5)
# plt.ylabel('Log[Ebreak]')

# plt.subplot(324)
# plt.plot(steps,par10,color='k',alpha=0.5)
# plt.ylabel('p1')

# plt.subplot(325)
# plt.plot(steps,par11,color='k',alpha=0.5)
# plt.ylabel('p2')

# plt.subplot(326)
# plt.plot(steps,par12,color='k',alpha=0.5)
# plt.ylabel('dist')

# fig2.tight_layout(h_pad=0.0)
# plt.savefig('space-covered2.jpg')


command=mcmc.runmodel(pars[ndx])
print(command)
ExtProcess=subprocess.Popen(command,shell=True)
ExtProcess.wait()

photfreqlist,fluxdensarr,dyninfo=mcmc.loadfits(pars[ndx])

fluxdens = {'freq1':0.327,'flux1':7.3, 'flux1err':0.7,'freq2':1.43,'flux2':7.,'flux2err':0.4,
            'freq3':4.8,'flux3':6.54,'flux3err':0.37,'freq4':70.,'flux4':4.3,'flux4err':0.6,
            'freq5':84.2,'flux5':3.94,'flux5err':0.7,'freq6':90.7,'flux6':3.8,'flux6err':0.4,
            'freq7':94.,'flux7':3.5,'flux7err':0.4,'freq8':100.,'flux8':2.7,'flux8err':0.5,
            'freq9':141.9,'flux9':2.5,'flux9err':1.2,'freq10':143.,'flux10':3.0,'flux10err':0.4}
#Radio
obsfreq=[]
obsflux=[]
obsfluxerr=[]
for i in range(1,11):
    obsfreq.append(fluxdens['freq'+str(i)]*1.e9)
    obsflux.append(fluxdens['flux'+str(i)]*fluxdens['freq'+str(i)]*1.e9/(1.e23))
    obsfluxerr.append(fluxdens['flux'+str(i)+'err']*fluxdens['freq'+str(i)]*1.e9/(1.e23)) 

#X-ray
obsxray={'minfreq':15.e3/hev, 'maxfreq':50.e3/hev, 'gamma':2.093, 'gammaerr':0.008,
        'flux':5.11e-11, 'fluxerr':0.05e-11} 

obsgamma=obsxray['gamma']
obsxrayflux=obsxray['flux']
xemin=obsxray['minfreq']*hcgs #erg
xemax=obsxray['maxfreq']*hcgs #erg
keverg=1.e3/erg #erg

term1=1./(2.-obsxray['gamma'])*(xemax**(2.-obsxray['gamma'])-xemin**(2.-obsxray['gamma']))
xnorm1=obsxray['flux']/term1*keverg**(-1.*obsxray['gamma'])

logfluxdens15kev1=(np.log10(xnorm1)-1.*obsxray['gamma']*
                (np.log10(xemin/keverg))+np.log10(xemin)+np.log10(hcgs))

logfluxdens50kev1=(np.log10(xnorm1)-1.*obsxray['gamma']*
                (np.log10(xemax/keverg))+np.log10(xemax)+np.log10(hcgs))

term2=1/(2-obsxray['gamma'])*(xemax**(2-obsxray['gamma'])-xemin**(2-obsxray['gamma']))
xnorm2=(obsxray['flux']+obsxray['fluxerr'])/term2*keverg**(-1*obsxray['gamma'])


logfluxdens15kev2=(np.log10(xnorm2)-1.*obsxray['gamma']*
                (np.log10(xemin/keverg))+np.log10(xemin)+np.log10(hcgs))

logfluxdens50kev2=(np.log10(xnorm2)-1.*obsxray['gamma']*
                (np.log10(xemax/keverg))+np.log10(xemax)+np.log10(hcgs))

obsxray_x1 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y1 = [obsxray['minfreq']*10**(logfluxdens15kev1),obsxray['maxfreq']*10**(logfluxdens50kev1)]

obsxray_x2 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y2 = [obsxray['minfreq']*10**(logfluxdens15kev2),obsxray['maxfreq']*10**(logfluxdens50kev2)]


gamma_pl=obsxray['gamma']+obsxray['gammaerr']
term3=1/(2-gamma_pl)*(xemax**(2-gamma_pl)-xemin**(2-gamma_pl))
xnorm3=(obsxray['flux']+obsxray['fluxerr'])/term3*keverg**(-1*gamma_pl)
#xnorm=obsxray['flux']/np.log(xemax/xemin)*keverg**(-1.*obsgamma)
#xnorm=obsxray['flux']/np.log10(xemax/xemin)*keverg**(-1.*obsxray['gamma']) #photons/ergs/cm^2/s

logfluxdens15kev3=(np.log10(xnorm3)-1.*gamma_pl*
                (np.log10(xemin/keverg))+np.log10(xemin)+np.log10(hcgs))

logfluxdens50kev3=(np.log10(xnorm3)-1.*gamma_pl*
                (np.log10(xemax/keverg))+np.log10(xemax)+np.log10(hcgs))



obsxray_x1 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y1 = [obsxray['minfreq']*10**(logfluxdens15kev1),obsxray['maxfreq']*10**(logfluxdens50kev1)]

obsxray_x2 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y2 = [obsxray['minfreq']*10**(logfluxdens15kev2),obsxray['maxfreq']*10**(logfluxdens50kev2)]

obsxray_x3 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y3 = [obsxray['minfreq']*10**(logfluxdens15kev3),obsxray['maxfreq']*10**(logfluxdens50kev3)]


#Gamma-ray
obsgammaray={'minfreq':0.2e+12/hev,'maxfreq':5.e+12/hev,'gamma':2.08,'gammaerr':0.22,
'freq':1.e+12/hev,'photdens':4.59e-13,'photdenserr':1.e-13}

gemin=obsgammaray['minfreq']*hcgs #erg
gemax=obsgammaray['maxfreq']*hcgs #erg

gfluxdens1=mcmc.phot2fluxdens(obsgammaray['photdens'],12,1e12/hev) #erg/s/cm^2/Hz
#gfluxdenserr=mcmc.phot2fluxdens(obsgammaray['photdens']+obsgammaray['photdenserr'],12,1e12/hev)-gfluxdens1
gfluxdenserr=mcmc.phot2fluxdens(obsgammaray['photdenserr'],12,1e12/hev)

print(photfreqlist)
photspeclist=[]
for i in range(len(photfreqlist)):
    temp_f=fluxdensarr[i]*photfreqlist[i]
    photspeclist.append(temp_f)

fig1=plt.figure()
fig1.set_size_inches(16,9)
#ax1=fig.add_subplot(111)
plt.plot(photfreqlist,photspeclist)
plt.scatter(obsfreq,obsflux,color='red',s=10)
plt.errorbar(obsfreq,obsflux,yerr=obsfluxerr,ls='none',color='red')
plt.errorbar(obsgammaray['freq'],gfluxdens1*obsgammaray['freq'],
    yerr=gfluxdenserr*obsgammaray['freq'],ls='none',color='purple')
plt.scatter(obsgammaray['freq'],gfluxdens1*obsgammaray['freq'],color='purple')
plt.plot(obsxray_x1,obsxray_y1,color='green')
plt.plot(obsxray_x2,obsxray_y2,color='green')
plt.xlabel(r'Frequency $\nu$ [Hz]')
plt.ylabel(r'$\nu F_{\nu}$ [ergs s$^{-1}$ cm $^{-2}$]')
plt.title('Photon Spectrum')
plt.xscale('log')
plt.yscale('log')
plt.savefig('PhotonSpectrum.png',dpi=200)
plt.show()
