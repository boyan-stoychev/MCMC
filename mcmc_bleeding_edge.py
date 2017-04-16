"""
############################################
ver 1.2:

-> Fixing E_break according to observations
-> Include both X-ray photon indices


############################################
ver 1.1: 

-> Linear interpolation for flux density and flux, instead of "nearest bin" to improve accuracy
-> Fit for radio, for hard x-ray, gamma-ray and expansion rate
-> To do - Rewrite functions to minimize rounding errors

############################################
ver 1.0:

-> Changed working folders on dalma from /home/ to /scratch/, fixed a couple of bugs
-> Fitting for pwn radius, soft&hard x-ray, and radio fluxes


############################################ 
"""


import numpy as np
import emcee
from emcee.utils import MPIPool
from mpi4py import MPI
from astropy.io import fits
import subprocess
import warnings
import sys
warnings.filterwarnings('ignore')

log=open('log.txt','w')
log.close()


log=open('log.txt','a')
# Physical constants:
yr = 31556926.0               #  s 
psc = 3.09e18                 #  cm 
msolar = 1.99e+33              #  g 
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
hev = 4.135667662e-15   #  Planck's constant[eV.s]
herg = 6.626068e-27       #  Planck's constant [ergs.s]
erg = 6.424e+11          #  Erg to eV[eV/erg]
kpc = 3.086e+21               #  kpc to cm

#Initial Input Parameters (Some from papers, some from G54.1+0.3 as starting point)
esn_51 = 10.  
mej = 10. #Solar Masses
nism = 0.001286 #G54.1+0.3
brakind = 2.7
age = 870.
charage=4850.
tau =  2*charage/(brakind-1)-age
edot_37 = 3.37
e0dot_37 = edot_37*(1+age/tau)**((brakind+1)/(brakind-1))
velpsr = 0
etag = 0 #fraction of SDL lost as radiation
etab = 0.0007 #Magnetization of Pulsar Wind
emin = 2.e-6 #GeV
ebreak = 9.7e-6 #GeV
emax = 4.5e-5 #GeV
p1 = 1.5 #Camilo et al, 2006
p2 = 3.   #Camilo et al, 2006
f_max = 0 #fraction of pulsar wind particle energy in Maxwellian component
kT_max = 0 #Energy of Maxwellian component
nic = 0 #Background photon fields in addition to CMB
dist_global = 5.




pwnpars=[np.log10(esn_51), np.log10(mej), np.log10(nism), brakind,
np.log10(tau), np.log10(age), np.log10(e0dot_37), velpsr, np.log10(etag), 
np.log10(etab), np.log10(emin), np.log10(emax), np.log10(ebreak), p1, p2, 
np.log10(f_max), np.log10(kT_max), nic,dist_global]




pwninput = []
for i in range(len(pwnpars)):
    if i<5 or (i>8 and i<15):
        pwninput.append(pwnpars[i])
pwninput.append(dist_global)
pwninput = np.asarray(pwninput)
#print(pwninput)

#Guess for starting point

#              log[Esn_51]       log[Mej]       log[Nism]        p           log[tau]       log[etab]
# pwninput = [-1.98312426941, 0.428001547411, -2.71746248658, 3.47308520696, 3.49940612028, -3.94923059876,
#            -1.65747519864, 8.34363511004, 1.4647924208, 0.207801264827, 2.59547538542, 1.61832545979]
#              log[Emin]       log[Emax]     log[ebreak]        p1             p2             dist

#             log[Esn_51]       log[Mej]       log[Nism]        p           log[tau]      
pwninput = [-1.98312426941, 0.428001547411, -2.71746248658, 3.47308520696, 3.49940612028, 
-3.94923059876, np.log10(emin),  np.log10(emax), 0.207801264827, 2.59547538542, 1.61832545979]
#  log[etab]      log[Emin]       log[Emax]      p1             p2             dist



#OUTPUT: fluxdensarr [erg/(s cm^2 Hz)]
def loadfits(pars):
    global badpars, photfreq
    badpars=False
    dist=pars[-1]
    rank = pool.rank
    elecfile = fits.open('/scratch/bks311/astro/code/'+str(rank)+'/modelres.elecspec.fits',ignore_missing_end=True)
    photfile = fits.open('/scratch/bks311/astro/code/'+str(rank)+'/modelres.photspec.fits',ignore_missing_end=True)
    dyninfofile = fits.open('/scratch/bks311/astro/code/'+str(rank)+'/modelres.dyninfo.fits',ignore_missing_end=True)
    try:
        dyninfo=dyninfofile[1].data
        photfreq=photfile[1].data
        photspec=photfile[2].data

        #elecnrg=elecfile[1].data
        #elecspec=elecfile[2].data

        #photfreq[1] - min 
        #photfreq[2] - max 
        #photfreq[3] - avg 
        #photfreq[4] - width

        photfreqarr=np.empty(len(photfreq))
        fluxdensarr=np.empty(len(photfreq))
        
        #elecnrglist=np.empty(len(photfreq))
        #elecspeclist=np.empty(len(photfreq))
        
        for i in range(len(photfreq)): 
            #elecnrglist[i] = elecnrg[i][3]
            #elecspeclist[i] = elecspec[0][i+1]
            
            if np.isnan(photspec[0][i+1]) == False and np.isnan(photfreq[i][3]) == False and photfreq[i][3]!=0:
                photfreqarr[i]=photfreq[i][3]
                temp2=photspec[0][i+1]*1e40/(4*np.pi*(dist*kpc)**2)
                fluxdensarr[i]=temp2


        elecfile.close()
        photfile.close()
        dyninfofile.close()


        return photfreqarr,fluxdensarr,dyninfo
    except:
        badpars=True
        return 0.,0.,0.
        log.write('Loadfits error')

# Input - Hz
# Output - erg/(cm^2 s Hz)
# Function uses linear interpolation 
def calcfluxdens(freq): 
    nbins=0
    try:
        for i in range(len(photfreqarr)):
            if freq>=photfreq[i][1] and freq<=photfreq[i][2]:

                fluxdens=(fluxdensarr[i]+(fluxdensarr[i+1]-fluxdensarr[i-1])/
                    (photfreqarr[i+1]-photfreqarr[i-1])*(freq-photfreqarr[i]))
                nbins+=1


        if nbins==1:
            return fluxdens
        else:
            return 0
    except ValueError:
        return -np.inf
        log.write('Calcfluxdens error')

# Input - Photdens -> [Photons/(cm^2 s (10^<power>eV)] ; freq -> Hz
# Output - [ergs/(s cm^2 Hz)]
def phot2fluxdens(photdens,power,freq):
    try:
        logphotnrg=np.log10(hcgs)+np.log10(freq) #erg
        logfluxdens=np.log10(photdens)+logphotnrg-power+np.log10(hev) #
        fluxdens=10**(logfluxdens)
        return fluxdens
    except AttributeError:
        log.write('Phot2fluxdens error (AttributeError)')
        return -np.inf

# Input - [eV]
# Output - [erg/(s cm^2)]
def calcflux(minenergy,maxenergy): 
    global flux
    #photfile = fits.open('/scratch/bks311/astro/code/'+str(rank)+'/modelres.photspec.fits',ignore_missing_end=True)
    #photfreq=photfile[1].data

    flux=0.
    minfreq = minenergy/hev
    maxfreq= maxenergy/hev
    freq=[]
    try:
        for i in range(len(photfreqarr)):
            if photfreq[i][2]>=minfreq and photfreq[i][1]<=maxfreq:

                minbinfreq=max(photfreq[i][1],minfreq)
                maxbinfreq=min(photfreq[i][2],maxfreq)

                #minbinfreq=photfreq[i][1]
                #maxbinfreq=photfreq[i][2]

                freqwidth=maxbinfreq-minbinfreq

                #avgbinfreq=(maxbinfreq+minbinfreq)/2
                #fluxdens=calcfluxdens(photfreqarr[i])

                fluxdens=fluxdensarr[i]*freqwidth
                flux+=fluxdens
            elif photfreq[i][1]>maxfreq:
                break
        return flux
    except:
        return np.inf
        log.write('Calcflux error')

# Input -> [eV]
# Output -> Photon index, Photon index error [Unitless]
def calcphotindex(minenergy,maxenergy): #input in eV
    #photfreqlist,photspeclist,dyninfo = loadfits(pars)
    minfreq = minenergy/hev
    maxfreq= maxenergy/hev
    photenergy=[]
    modphotdens=[]
    for i in range(len(photfreqarr)):
        if photfreqarr[i]>=minfreq and photfreqarr[i]<=maxfreq:
            logphotenergy=np.log10(herg) + np.log10(photfreqarr[i]) #erg
            photenergy.append(10**(logphotenergy)*erg*10**(-3)) #keV
            logmodphotdens=np.log10(fluxdensarr[i])+3*-np.log(hev)-logphotenergy
            modphotdens.append(10**(logmodphotdens))
    try:
        photinfo,err,_,_,_=np.polyfit(np.log10(photenergy),np.log10(modphotdens),1,full=True)
        gamma = -1*photinfo[0]
        gammaerr = err[0]
        return gamma,gammaerr
    except:
        return np.inf,1
        log.write('Calcphotindex error')

# Input -> [eV]
# Output -> [photons/(s cm^2 eV^<power>)]
def calcphotdens(energy,power): 
    nbins=0

    try:
        freq=energy/hev
        for i in range(len(photfreqarr)):
            if freq>=photfreq[i][1] and freq<=photfreq[i][2]:      
                nbins+=1      
                fluxdens=calcfluxdens(freq) #erg/(s cm^2 Hz)
                logphotenergy=np.log10(hcgs)+np.log10(photfreqarr[i]) # erg
                logphotdens=np.log10(fluxdens)+power-np.log10(hev)-logphotenergy
                photdens=10**logphotdens
        if nbins==1:
            return photdens
    except:
        return -np.inf
        log.write('Calcphotdens error')




def calcrad(pars): #in arcminutes
    dist = pars[-1] #kpc
    Rpwn = dyninfo['RADPWN']  #parsec
    #Rsnr = dyninfo['RADSNR']
    #thetaSNR = np.rad2deg(np.arcsin(Rsnr/(dist*10**3)))*60
    thetaPWN = float(np.rad2deg(np.arcsin(Rpwn/(dist*10**3)))*3600)#arcseconds
    return thetaPWN
    
def runmodel(pars):
    strtstep='-1'
    stresn=str(10**pars[0])
    strmej=str(10**pars[1])
    strn=str(10**pars[2])
    strp=str(pars[3])
    strtau=str(10**pars[4])
    #Set age and e0dot based on tau and p
    age=2*charage/(pars[3]-1)-10**pars[4]
    strage=str(age)
    stre0=str(edot_37*(1+age/(10**pars[4]))**((pars[3]+1)/(pars[3]-1)))
    strvelpsr=str(pwnpars[7])
    if pwnpars[8] < -1.e4 or np.isfinite(pwnpars[8]) == False:
        streitag = '0'
    else:
        streitag = str(10**pwnpars[8])
    streitab=str(10**pars[5])
    stremin=str(10**pars[6])
    stremax=str(10**pars[7])
    #strebreak=str(10**pars[8])   #-> Use when fitting for Ebreak
    strebreak=str(ebreak)
    strinjinx1=str(pars[9])
    strinjinx2=str(pars[10])
    if pwnpars[15] < -1.e4:
        strfmax='0'
        strktmax='0'
    else:
        strfmax = str(10**pwnpars[15])
        strktmax = str(10**pwnpars[16])
    strnic = str(pwnpars[17])
    nic = pwnpars[17]
    if nic>0:
        strictemp=''
        stricnorm=''
        index=17
        for i in range(nic):
            index+=1
            strictemp = str(strictemp+' '+str(10**pwnpars[index]))
        for i in range(nic):
            index+=1
            stricnorm = str(stricnorm+' '+str(10**pwnpars[index]))
        command=('../pwnmodel.exe '+ strtstep +' '+ stresn+ ' '+ strmej +
                  ' '+strn+' '+strp+' '+strtau+' '+strage+' '+
                  stre0+' '+strvelpsr+' '+streitag+' '+streitab +' '+
                  stremin+' '+stremax+' '+strebreak+' '+strinjinx1 +' '+
                  strinjinx2+' '+strfmax+' '+strktmax+' '+strnic+' '+strictemp+' '
                  +stricnorm) 

    else:
        command=('../pwnmodel.exe '+ strtstep +' '+ stresn+ ' '+ strmej +
                      ' '+strn+' '+strp+' '+strtau+' '+strage+' '+
                      stre0+' '+strvelpsr+' '+streitag+' '+streitab +' '+
                      stremin+' '+stremax+' '+strebreak+' '+strinjinx1 +' '+
                      strinjinx2+' '+strfmax+' '+strktmax+' '+strnic)

    return command

def prior(pars): #Parameter constraints go here:
    logmaxtau=np.log10(2.*charage/(pars[3]-1)-0.5)
    if (pars[9]<0 or pars[10]<0 or pars[2]<-6 or pars[0]<-3 or pars[0]>4 or pars[1]>2 or pars[6]<-3.3 
     or pars[2]>-0.1 or pars[6]>pars[7] or pars[6]>pars[8] or pars[7]<pars[8] or pars[5]>0 or pars[11]<0
     or pars[3]<1 or pars[3]>5 or pars[4]>logmaxtau):
        #print('Parameter check failed, returning -inf')
        return -np.inf
        log.write('Parameter check failed')
    else:
        return 0

    
    
def pwnevol(pars): # Use MPI pool to execute model as a subprocess on each core, read .fits files and 
                   # calculate likelihood for set of parameters.
    global photfreqarr,fluxdensarr,dyninfo
    rank = pool.rank

    #Check parameter constraints
    pr = prior(pars)
    if np.isfinite(pr)==False:
        return -np.inf,'bad'
    #Run model, wait for it to finish
    else:
        command = runmodel(pars)
        ExtProcess = subprocess.Popen(command, cwd = '/scratch/bks311/astro/code/'+str(rank)
            ,shell=True,stdout=log,stderr=log)
        ExtProcess.wait()

        #Parameter/file output check
        photfreqarr,fluxdensarr,dyninfo=loadfits(pars)
        if badpars==True:
            return -np.inf,'bad'
        else:
            dist = pars[len(pars)-1]
            npwnpars = len(pars)
            strpwnpars=''
            for par in pars:
                strpar = str(par)
                strpwnpars = strpwnpars+strpar+' '
            
            chi2 = 0
            npar = 0
            strmodpred=''
            #obsthetapwn = 40 #[arcmin]
            
        #    #PWN radius
        #    radpwn = dyninfo['RADPWN']
        #    thetapwn, thetasnr = Radii() 
        #    npar+=1
        #    logthetapwn = np.log10(thetapwn)
        #    strobs = strobs+''+str(thetapwn)
        #    #chi2 += 
        #    
        #    #SNR radius
        #    radsnr = dyninfo['RADSNR']
        #    logthetasnr - np.log10(thetapwn)
        #    #chi2 +=
        #    npar+=1
        #    strobs=strobs+''+str(thetasnr)
            
            #Radio
            for i in range(1,11):
                obsfreq = fluxdens['freq'+str(i)]*1e9 #in Hz
                obsfluxdens = fluxdens['flux'+str(i)]
                obsfluxdenserr = fluxdens['flux'+str(i)+'err']
                modfluxdens = calcfluxdens(obsfreq)*1e23 #Jy
                chi2 += ((modfluxdens-obsfluxdens)/obsfluxdenserr)**2
                npar+=1
                strmodpred = strmodpred+' '+str(modfluxdens) #Predicted Observables
            

            modthetapwn=calcrad(pars)
            chi2 += ((modthetapwn-obsthetapwn)/obsthetapwnerr)**2
            npar+=1
            strmodpred=strmodpred+' '+str(modthetapwn)

            vpwn = dyninfo['VPWN'] # km/s
            rpwn = dyninfo['RADPWN'] # parsecs
            modexprate = vpwn/rpwn/(psc*1e-5)*(3.154e7)*100 # %/yr
            chi2 += ((modexprate-obsexprate)/obsexprateerr)**2
            npar+=1
            strmodpred=strmodpred+' '+str(modexprate)

            #modgamm1,modgammerr1 = calcphotindex(3.e3,10.e3)
            modgamm2,modgammerr2 = calcphotindex(15e3,50e3)
            #chi2 += ((modgamm1-obsgamm1)/obsgammerr1)**2
            chi2 += ((modgamm2-obsgamm2)/obsgammerr2)**2
            strmodpred = strmodpred+' '+str(modgamm2)
            npar+=2

            log.write(str(chi2))

            modgamm3,modgammerr3 = calcphotindex(2e11,5e12)
            chi2 += ((modgamm3-obsgamm3)/obsgammerr3)**2
            npar+=1
            strmodpred=strmodpred+' '+str(modgamm3)
            #modflux1=calcflux(2.e3,8.e3)
            modflux2=calcflux(15e3,50e3)

            #chi2 += ((modflux1-F2_8)/F2_8err)**2
            chi2 += ((modflux2-F15_50)/F15_50err)**2
            npar+=2
            strmodpred=strmodpred+' '+str(modflux2)

            #Photon density fit is messed up for some reason, RIP
            modphotdens=calcphotdens(1.e12,12) # photon density in photons/cm^2/s/TeV
            obsphotdens=photdens['dens1']
            photdenserr=photdens['dens1err']

            #modgfluxdens1=calcfluxdens(1e12/hev)
            #modgfluxdens2=phot2fluxdens(modphotdens,12,1e12/hev)
            #print('Gamma fluxdens from calcfluxdens is: '+str(modgfluxdens1))
            #print('Gamma fluxdens from phot2fluxdens is: '+str(modgfluxdens2))
            obsgfluxdens=phot2fluxdens(obsphotdens,12,1e12/hev)
            obsgfluxdenserr=phot2fluxdens(photdenserr,12,1e12/hev)
            try:
                chi2 += ((modphotdens-obsphotdens)/photdenserr)**2
                #chi2 += ((modgfluxdens1-obsgfluxdens)/obsgfluxdenserr)**2

            except TypeError:
                chi2=np.inf

            strmodpred=strmodpred+' '+str(modphotdens)# +' '+str(modfluxdens2)

            lnprob=-0.5*chi2
            strchi2 = str(chi2)
            strlnprob = str(lnprob)
                
            if np.isnan(lnprob) == True:
                print('NAN check failed, returning -inf')
                return -np.inf,'bad'
                log.write('lnprob is nan')

            else:
                return lnprob,strmodpred
        
        
   #print(radpwn)


# OBSERVABLES

#SNR properties
#obsthetasnr={'rad':2/3} #[arcmin]

#PWN properties
obsthetapwn, obsthetapwnerr = 40., 4. # arcsec
obsexprate, obsexprateerr=0.11,0.02   # %/yr


#Flux Densities
fluxdens = {'freq1':0.327,'flux1':7.3, 'flux1err':0.7,'freq2':1.43,'flux2':7.,'flux2err':0.4,
            'freq3':4.8,'flux3':6.54,'flux3err':0.37,'freq4':70.,'flux4':4.3,'flux4err':0.6,
            'freq5':84.2,'flux5':3.94,'flux5err':0.7,'freq6':90.7,'flux6':3.8,'flux6err':0.4,
            'freq7':94.,'flux7':3.5,'flux7err':0.4,'freq8':100.,'flux8':2.7,'flux8err':0.5,
            'freq9':141.9,'flux9':2.5,'flux9err':1.2,'freq10':143.,'flux10':3.0,'flux10err':0.4} # freq in GHz, flux in Jy
#Unabsorbed Flux
F2_8 = 5.27e-11 #ergs/(s*cm^2)
F2_8err = 0.05e-11

F15_50 = 5.11e-11
F15_50err = 0.05e-11


#Photon Index
obsgamm1,obsgammerr1 = 1.996, 0.008 #3-10 keV 
obsgamm2,obsgammerr2 = 2.093, 0.008 #10-45 keV 

obsgamm3,obsgammerr3 = 2.08,0.22 #0.2 - 5 TeV 



photdens= {'nrg1':1.e12,'dens1':4.59e-13,'dens1err':1.0e-13}


#MPI pool for parallelization
try:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)



    nwalkers = 400
    nsteps=4000
    ndim = len(pwninput)

    #Initialize Walkers from normal distribution around values found in pwninput
    p0 = np.random.rand(nwalkers,ndim) 
    for i in range(nwalkers):
        for j in range(ndim):
            h = np.random.normal(pwninput[j],0.01)
            p0[i][j] = h

    sampler = emcee.EnsembleSampler(nwalkers,ndim,pwnevol,pool=pool,live_dangerously=True)
    f1 = open("chain.dat", "w")
    f1.close()
    f2 = open("prob.dat","w")
    f2.close()
    f3 = open('predictobs.dat','w')
    f3.close()

    #Run MCMC and store progress incrementaly
    for result in sampler.sample(p0, iterations=nsteps, storechain=False):
        position = result[0]
        prob=result[1]
        predictobs=result[3]
        f1 = open("chain.dat", "a")
        f2 = open("prob.dat","a")
        f3 = open("predictobs.dat",'a')
        for k in range(position.shape[0]):
            f1.write(" ".join(str(i) for i in position[k])+'\n')
            f2.write(str(prob[k])+'\n')
            f3.write(predictobs[k]+'\n')
        f1.close()
        f2.close()
        f3.close()


    pool.close()
    log.close()

except ValueError:
    pool=0