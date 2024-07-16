import os
from astropy import units as u
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import shutil
import string
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from numpy import median
import astropy.constants as c
from astropy import constants as const
#############################
#constants
c_light=299792.458
LYA = 1215.6 # angstrom

############################
def getVel2(wave,l0):
    vel=const.c.cgs.value*1e-5*(wave-l0)/(l0) #in km/s
    return vel
#############################
def getVel(wave,l0,zabs):
    wave_center=l0*(zabs + 1.)
    vel=const.c.cgs.value*1e-5*(wave**2-wave_center**2)/(wave**2+wave_center**2) #in km/s
    return vel
#############################

def findV(zAbs, zEm):
    c_light=299792.458
    v = ((((1.0+zEm)**2) - ((1.0+zAbs)**2))/(((1.0+zEm)**2) + ((1.0+zAbs)**2))) * c_light
    return v
##############################
def findZAbs(v, zEm):
    c = 299792.458
    temp = ((1.0+zEm)**2)*((c-v)/(c+v))
    zAbs = math.sqrt(temp) - 1.0
    return zAbs
##############################

#########################################################################
###############################
# velocity widths for a particular ion
def widths(i,j,ion,transition):
    return abs(max(spec[i:j][ion+transition+'_Vel'])-min(spec[i:j][ion+transition+'_Vel']))
vwidths=np.vectorize(widths)
###############################
def getEW(d,zabs):
    d = d.reset_index(drop=True)
    ew=np.trapz(1-d['FLUX'],x=d['WAVELENGTH'])/(1+zabs)
    d['Del-W']=d[['WAVELENGTH']].shift(-1)-d[['WAVELENGTH']]
    err=(sum((d['ERROR']*d['Del-W'])**2))**0.5/(1+zabs)
    return(ew,err)
##############################

# function to translate the observed wavelength to rest wavelength
def restwavl(w,z):
    x=w/(1.0+z)
    return (x)
#############################

#########################################################
def fivesigmadet(x,y):
    try:
        val=min(spec[x:y]['FLUX'][spec[x:y]['FLUX']>0])
        loc_min=np.where(spec['FLUX']==val)[0][0]
        if (1-val)/spec['ERROR'][loc_min] >= 5:
            return True
        else:
            return False
    except:
        return False
#########################################################    


##########functions for linedetector######################
def speciesinterest(plot_ions,transition_library,choose):
    """
    Return species of interest in plot_ions. Omit transitions not in choose for a corresponding ion.
    """
# Transition Wavelengths and their power/noise arrays
    from collections import OrderedDict
    species = {}
    power = {}
    noise = {}
    working = {}
    sigma = {}

    # Loop through atoms.dat and match with search_ions
    # Initialize arrays to search for transitions
    strongest_trans = OrderedDict()
    second_trans = OrderedDict()
    for search_trans in plot_ions:
        species[search_trans] = OrderedDict()
        max_oscillator = 0.
        m1 = m2 = float('-inf')
        oldwav1 = oldwav2 = " "
        for transition in transition_library:
            [atom,tranwav] = transition[0].split("_")
            current_oscillator = transition[3]
            gamma = transition[4]
            atomic_mass = transition[5]
            # Initialize arrays
            if ( atom == search_trans ):
                species[atom][tranwav] = transition[2] , current_oscillator, atomic_mass, gamma
                power[tranwav] = []
                noise[tranwav] = []
                working[tranwav] = []
                sigma[tranwav] = []
                # Find strongest and 2nd strongest transitions (e.g. MgII2796 is strongest MgII transition and MgII2803 is the 2nd)
                if ( transition[3] > m2 ):
                    if ( transition[3] >= m1 ):
                        m1,m2 = transition[3], m1
                        oldwav1,oldwav2 = tranwav, oldwav1
                        strongest_trans[search_trans],second_trans[search_trans] = "{}".format(tranwav),"{}".format(oldwav1)
                    else:
                        m2 = transition[3]
                        oldwav2 = tranwav
                        second_trans[search_trans] = "{}".format(tranwav)
        second_trans[search_trans] = "{}".format(oldwav2)
    if choose != None:
        for var in choose:
            for transition in species[var].copy().keys():
                if transition not in choose[var]:
                    species[var].pop(transition)    


    return species


def blue_red_limits(zem,wavelength,species,flag=1):
#Search redward of the Lya emission
    if flag==0:
        if ( (LYA * (1. + zem)) > wavelength[0]):
            blue_limit = LYA * (1. + zem)
        else:
            blue_limit = wavelength[0]
        # Emission redshift of quasar - 3000 km/s
        emission_limit = 1. + zem - 0.01
        red_limit = {}
        for specie in species:
            red_limit[specie] = []
            # Choose either the reddest_transition or the end of the spectrum
            reddest = max(species[specie])
            if ((species[specie][reddest][0] * emission_limit) < wavelength.iloc[-1]):
                red_limit[specie] = species[specie][reddest][0] * emission_limit
            else:
                red_limit[specie] = wavelength.iloc[-1]
        max_red = max(red_limit.values())
    else:
        blue_limit=wavelength.iloc[0]
        max_red=wavelength.iloc[-1]
    return blue_limit, max_red

def fluxselector(spec,blue_limit,max_red):
    spec = spec[(spec['WAVELENGTH'] >= blue_limit) & (spec['WAVELENGTH'] <= max_red)]
    return spec

def fluxcleanup(spec,telluric_windows=[(9300.,9630.),(7594.,7700.),(6868.,6932.),(6277.,6318.)]):
    # Clean cosmic rays/bad high flux values
    spec['ERROR'][spec['FLUX'] > 1.5] = 0.
    spec['FLUX'][spec['FLUX'] > 1.5] = 1.
    # Avoid Telluric features, in order from strongest to weakest
    # 9300 - 9630 feature
    for i in telluric_windows:
        spec['FLUX'][(spec['WAVELENGTH']>i[0]) & (spec['WAVELENGTH']<i[1])] = 1.0
    # Remove negative errors        
    spec['FLUX'][spec['ERROR']<0.] = 1.
    spec['ERROR'][spec['ERROR']<0.] = 0.
    # Remove negative flux
    spec['ERROR'][spec['FLUX'] < 0.] = 0.
    spec['FLUX'][spec['FLUX'] < 0.] = 0.
    return spec

def absorptionlocator(spec):
    spec['tag'] = spec['FLUX'] < 1.0 
    spec['mask'] = np.where(spec['tag'],1,0)
    fst = spec.index[spec['tag'] & ~ spec['tag'].shift(1).fillna(False)]
    lst = spec.index[spec['tag'] & ~ spec['tag'].shift(-1).fillna(False)]
    pr = [(i, j) for i, j in zip(fst, lst) if j > i+4]
    return pr


def addvel2spec(spec,species):
    for specie,transitions in species.items():
        for transition in transitions.items():
            spec[specie+transition[0]+'_Vel']=c_light*(spec['Rest-Wavelength']**2-transition[1][0]**2)/(spec['Rest-Wavelength']**2+transition[1][0]**2)
    return spec

def widths(i,j,ion,transition,spec):
    return abs(max(spec[i:j][ion+transition+'_Vel'])-min(spec[i:j][ion+transition+'_Vel']))
vwidths=np.vectorize(widths)   



def getINFO(number,ion,transition,spec,species,zem,pr):
    
    d=spec[(spec['Rest-Wavelength']>=spec['Rest-Wavelength'][np.array(pr)[number][0]]) &
       (spec['Rest-Wavelength']<=spec['Rest-Wavelength'][np.array(pr)[number][1]])]
    
    #identify absorption lines within this region by determining the minima
    if len(d['FLUX'])>=5:
        x=savgol_filter(d['FLUX'],5,1) #smoothing the array
    else:
        x=savgol_filter(d['FLUX'],3,1) #smoothing the array
    
    yvals=x[argrelextrema(x, np.less)[0]]
    xvals=np.asarray(d[ion+min(list(species[ion].keys()))+'_Vel'])[argrelextrema(x, np.less)[0]]
    
    k=list(range(0,40,1))    
    zlocs=[]
    
    list_species=list(species[ion].keys())
    list_species.remove(min(list(species[ion].keys())))
    
    for j in xvals:
        for i in k:
            try:
                for listspec in list_species:
                    if spec[ion+listspec+'_Vel'][np.array(pr)[number+i][0]] <= j <= spec[ion+listspec+'_Vel'][np.array(pr)[number+i][1]]:
                        zlocs.append(findZAbs(-j, zem))
            except IndexError:
                break

    d = d.reset_index(drop=True)
    try:
        if transition==min(species[ion]):
            return (min(d[ion+transition+'_Vel']),max(d[ion+transition+'_Vel']),median(d['WAVELENGTH'])/species[ion][transition][0]-1,
                    zlocs)
        else:
            return (min(d[ion+transition+'_Vel']),max(d[ion+transition+'_Vel']),median(d['WAVELENGTH'])/species[ion][transition][0]-1,
                    None)
    except:
        if transition==min(species[ion]):
            return (min(d[ion+transition+'_Vel']),max(d[ion+transition+'_Vel']),median(d['WAVELENGTH'])/species[ion][transition][0]-1,
                    None)
        else:
            return (min(d[ion+transition+'_Vel']),max(d[ion+transition+'_Vel']),median(d['WAVELENGTH'])/species[ion][transition][0]-1,
                    None)
#vgetINFO=np.vectorize(getINFO)

def getdf(spec,species,zem,pr,search_ions):
    df = pd.DataFrame([])
    for specie,transitions in species.items():
        if specie in search_ions:
            for transition in transitions:
                for i in range(len(pr)):
                    x=getINFO(i,specie,transition,spec,species,zem,pr)
                    if (x[3]!=[]) and (x[3]!=None) and (x[0]>=search_ions[specie][0]) and (x[1]<=search_ions[specie][1]):
                        d = {'SPECIES': specie,'TRANSITION':transition,'NUMBER':i,'VEL_MIN':x[0],'VEL_MAX':x[1],'MEDIAN-REDSHIFT':x[2],'REDSHIFT':[x[3]]}
                        df=df.append(pd.DataFrame(data=d),ignore_index=True)
    return df
