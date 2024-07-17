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

def clean_spectra(wave,flux,err):
    wave_c = wave[(err<0.05)&(flux>0)]
    flux_c = flux[(err<0.05)&(flux>0)]
    err_c = err[(err<0.05)&(flux>0)]

    return wave_c,flux_c,err_c
    
############## Functions from myfuncpyNorm ###############

def fix_unwriteable_spec(spec):
    # FIX NON-WRITEABLE ARRAYS due to discontiguous memory
    for kkk in spec.keys():
        if isinstance(spec[kkk],(np.ndarray)):
            spec[kkk] = spec[kkk].copy()

    return spec
def integration_weights(x,limits):
    # Calculate the weighting for each pixel in the column density integration.
    #  Includes partial pixel weighting for edge effects.

    # Pixel number array
    pix_num_array = np.arange(len(x))

    # Pixel spacing
    delx = np.median(np.roll(x,-1)-x)

    # Find the pixels that are fully within our integration range.
    idx = ((x-delx/2 >= limits[0]) & (x+delx/2 < limits[1]))
    weights = idx*1.0

    # Limits of complete pixels in indices.
    pix_limits = [np.min(pix_num_array[idx]),
                    np.max(pix_num_array[idx])]

    # Calculate fractions of edge pixels in the integration:
    # Identify edge pixels
    lo_pix = pix_limits[0]-1
    hi_pix = pix_limits[1]+1

    # Fraction of edge pixels contained within integration ranges
    lo_frac = ((x[lo_pix]+delx/2)-limits[0])/(delx)
    hi_frac = (limits[1]-(x[hi_pix]-delx/2))/(delx)

    # Assign fractional weights to the edge pixels
    weights[lo_pix] = lo_frac
    weights[hi_pix] = hi_frac

    return weights


def pynn_istat(spec_in,integration_limits = None,
                partial_pixels = True):

    spec = spec_in.copy()

    # FIX NON-WRITEABLE ARRAYS due to discontiguous
    # memory in some readsav inputs
    if ~spec['vel'].flags.writeable:
        spec = fix_unwriteable_spec(spec)

    # Make sure there are integration limits:
    if integration_limits is None:
        integration_limits = [spec['v1'],spec['v2']]

    # Some constants and flags
    column_factor = 2.654e-15
    ew_factor = 1.13e17
    lightspeed = 2.998e5 # km/s

    velocity = spec['vel'].copy()
    flux = spec['flux'].copy()
    flux_err = spec['eflux'].copy()
    wavc=spec['wavc']
    fval=spec['fval']

    # Deal with the continuum
    if "contin" in spec.keys():
        continuum=spec['contin'].copy()
    else:
        try:
            continuum=spec['cont'].copy()
        except:
            continuum=spec['ycon'].copy()

    if "contin_err" in spec.keys():
        continuum_err = spec['contin_err'].copy()
    else:
        try:
            continuum_err = spec['econt'].copy()
        except:
            continuum_err = spec['ycon_sig'].copy()


    # Define the limits of the integration:
    if not integration_limits:
        integration_limits = [spec['v1'],spec['v2']]

    # Work out partial pixel weighting
    weights = integration_weights(velocity,integration_limits)
    # Uniform weighting if not partial pixel weighting
    if not partial_pixels:
        weights = np.zeros_like(velocity)
        xlim1, xlim2 = xlimit(velocity,integration_limits)
        weights[xlim1:xlim2+1] = 1.0

    # An array of delta v:
    delv = velocity[1:]-velocity[:-1]
    delv = np.concatenate((delv,[delv[-1]]))

    # TODO: Saturation?
    # Calculate the zeroth moment
    tau = np.log(np.abs(continuum/flux))
    tau_tot = np.sum(tau*delv*weights)

    # M1
    # Calculate the first moment (average velocity)
    a = np.sum(tau*velocity*delv*weights)
    m1 = a/tau_tot

    # M1 error
    dadi = -1./flux*velocity*delv*weights
    dadc = 1 / continuum*velocity*delv*weights
    dwdi = -1./flux*delv*weights
    dwdc = 1 / continuum*delv*weights

    dm1di = (tau_tot * dadi - a * dwdi) / tau_tot**2
    dm1dc = (tau_tot * dadc - a * dwdc) / tau_tot**2

    q1 = np.sqrt(np.sum((flux_err*weights)**2 * dm1di**2))
    q2 = np.sum(np.sqrt((continuum_err*weights)**2 * dm1dc**2))

    m1err = np.sqrt(q1**2 + q2**2)

    # M2
    # Calculate the second moment (width)
    bsqared = np.sum(tau*(velocity-m1)**2*delv*weights)
    m2 = np.sqrt(bsqared/tau_tot)

    # M2 error
    dbdi = -1./flux*(velocity-m1)**2 * delv*weights
    dbdc = 1./ continuum*(velocity-m1)**2 * delv*weights
    dbdm1 = -2.*tau*(velocity-m1) * delv*weights

    dm2di = (tau_tot * dbdi - bsqared*dwdi) / tau_tot**2
    dm2dc = (tau_tot * dbdc - bsqared*dwdc) / tau_tot**2
    dm2dm1 = dbdm1 / tau_tot

    q1 = np.sqrt(np.sum((flux_err*weights)**2 * dm2di**2))
    q2 = np.sum(np.sqrt((continuum_err*weights)**2 * dm2dc**2))
    q3 = np.sqrt(np.sum(m1err**2 * dm2dm1**2))

    m2err = np.sqrt(q1**2 + q2**2 + q3**2)
    m2err = m2err / (2.*m2)

    bvalue = m2*np.sqrt(2.)
    bvalue_err = m2err*np.sqrt(2.)

    # M3
    # Calculate the third moment (skewness)
    c = np.sum(tau*weights*((velocity*weights - m1)/m2)**3*delv*weights)
    m3 = c/tau_tot

    # M3 error
    dfdi = -1./flux*((velocity*weights - m1) / m2)**3 * delv*weights
    dfdc = 1./continuum*((velocity*weights - m1) / m2)**3 * delv*weights
    dfdm1 = tau*weights*3.*((velocity*weights - m1) / m2)**2 * (-1./ m2) * delv*weights
    dfdm2 = tau*weights*3.*((velocity*weights - m1) / m2)**2 * (m1 - velocity*weights) / m2**2 * delv*weights

    dm3di = (tau_tot * dfdi - c * dwdi) / tau_tot**2
    dm3dc = (tau_tot * dfdc - c * dwdc) / tau_tot**2
    dm3dm1 = dfdm1 / tau_tot
    dm3dm2 = dfdm2 / tau_tot

    q1 = np.sqrt(np.sum((flux_err*weights)**2 * dm3di**2))
    q2 = np.sum(np.sqrt((continuum_err*weights)**2 * dm3dc**2))
    q3 = np.sqrt(np.sum(m1err**2 * dm3dm1**2))
    q4 = np.sqrt(np.sum(m2err**2 * dm3dm2**2))

    m3err = np.sqrt(q1**2 + q2**2 + q3**2 + q4**2)

    # Calculate the extent (same as m2, except that m1 is assumed to be 0)
    b4 = np.sum(tau*(velocity*weights - 0)**2*delv*weights)
    m4 = np.sqrt(b4/tau_tot)

    # M4 error
    dbdi = -1./flux*(velocity*weights - 0.0)**2 * delv*weights
    dbdc = 1./ continuum*(velocity*weights - 0.0)**2 * delv*weights

    dm4di = (tau_tot * dbdi - b4 * dwdi) / tau_tot**2
    dm4dc = (tau_tot * dbdc - b4 * dwdc) / tau_tot**2

    q1 = np.sqrt(np.sum((flux_err*weights)**2 * dm4di**2))
    q2 = np.sum(np.sqrt((continuum_err*weights)**2 * dm4dc**2))

    m4err = np.sqrt(q1**2 + q2**2)
    m4err = m4err / (2. * m4)


    # Velocities at 5% and 95% of total optical depth as dv90
    tau_cum = np.cumsum(tau*delv*weights)
    # 5% limit
    v90a = (np.abs(tau_cum/tau_tot-0.05)).argmin()
    v90a = velocity[v90a]
    # 95% limit
    v90b = (np.abs(tau_cum/tau_tot-0.95)).argmin()
    v90b = velocity[v90b]

    # Calculate dv90:
    dv90 = np.abs(v90b - v90a)


    # Fill the spec output
    spec['va'] = m1
    spec['va_err'] = m1err
    # spec['ba'] = m2
    # spec['ba_err'] = m2err
    spec['ba'] = bvalue
    spec['ba_err'] = bvalue_err
    spec['m3'] = m3
    spec['m3_err'] = m3err

    spec['dv90'] = dv90
    spec['v90a'] = v90a
    spec['v90b'] = v90b

    # Get rid of old-style keywords
    try:
        del spec['vaerr']
        del spec['baerr']
        del spec['m3err']
    except:
        pass

    return spec



def pynn_column(spec_in, integration_limits = None,
                partial_pixels = True):

    spec = spec_in.copy()

    # FIX NON-WRITEABLE ARRAYS due to discontiguous
    # memory in some readsav inputs
    if ~spec['vel'].flags.writeable:
        spec = fix_unwriteable_spec(spec)

    # Make sure there are integration limits:
    if integration_limits is None:
        integration_limits = [spec['v1'],spec['v2']]

    # Some constants and flags
    column_factor = 2.654e-15
    flag_sat = False

    velocity = spec['vel'].copy()
    flux = spec['flux'].copy()
    flux_err = spec['eflux'].copy()
    wavc=spec['wavc']
    fval=spec['fval']

    # Deal with the continuum:
    if "contin" in spec.keys():
        continuum=spec['contin'].copy()
    else:
        try:
            continuum=spec['cont'].copy()
        except:
            continuum=spec['ycon'].copy()

    if "contin_err" in spec.keys():
        continuum_err = spec['contin_err'].copy()
    else:
        try:
            continuum_err = spec['econt'].copy()
        except:
            continuum_err = spec['ycon_sig'].copy()

    # Work out partial pixel weighting
    weights = integration_weights(velocity,integration_limits)
    # Uniform weighting if not partial pixel weighting
    if not partial_pixels:
        weights = np.zeros_like(velocity)
        xlim1, xlim2 = xlimit(velocity,integration_limits)
        weights[xlim1:xlim2+1] = 1.0

    # An array of delta v:
    delv = velocity[1:]-velocity[:-1]
    delv = np.concatenate((delv,[delv[-1]]))

    # Test for clearly saturated pixels:
    #   -- If the idx_saturation is already filled, use the results:
    try:
        idx_saturation = ((flux <= 0.) | (idx_saturation == True))
    except:
        idx_saturation = (flux <= 0.)

    # Fix saturation if it's present.
    flux[idx_saturation] = np.abs(flux[idx_saturation])
    flux[(flux==0)] = 2.*flux_err[(flux==0)]

    # Set overall saturation flag for saturation in the integration range
    if (idx_saturation*weights).sum() > 0:
        flag_sat = True

    # Create an optical depth array and its error
    tau_array = np.log(continuum / flux)
    tau_array_err = np.sqrt((flux_err/flux)**2)

    # If optical depth NaN, set to zero.
    # This happens when continuum < 0.
    bd = np.isnan(tau_array)
    tau_array[bd] = 0.
    tau_array_err[bd] = 0.

    # TODO: Include an AOD array in output w/continuum errors.
    # Integrate the apparent optical depth
    tau_int = np.sum(tau_array*delv*weights)
    tau_int_err = \
     np.sqrt(np.sum((tau_array_err*delv*weights)**2))

    # Create an apparent column density array
    nav_array = tau_array/(wavc*fval*column_factor)
    nav_err_stat = tau_array_err/(wavc*fval*column_factor)
    nav_err_cont = (continuum_err/continuum)/(wavc*fval*column_factor)
    nav_err_tot = np.sqrt(nav_err_stat**2 + nav_err_cont**2)

    # Integrate the apparent column density profiles
    column = tau_int/(wavc*fval*column_factor)

    # Error in the column
    column_err = tau_int_err/(wavc*fval*column_factor)

    # Continuum error: errors are correlated, so don't add in quadrature.
    column_err_cont = \
        np.sum(((continuum_err/continuum)*delv*weights)) /\
         (wavc*fval*column_factor)

    # Background uncertainty -- Not applied
    # z_eps = 0.01  # Fractional bg error
    # yc1 = continuum*weights*(1.-z_eps)
    # y1  = flux*weights-continuum*weights*z_eps
    # tau1 = np.sum(np.log(yc1/y1)*delv*weights)
    # col1 = tau1 / (wavc*fval*column_factor)
    # column_err_zero = np.abs(col1-column)

    # Combine errors
    column_err_total = np.sqrt(column_err**2 \
        +column_err_cont**2)

    log_n_err_lo = np.log10(column-column_err_total) - np.log10(column)
    log_n_err_hi = np.log10(column+column_err_total) - np.log10(column)

    spec['v1'] = integration_limits[0]
    spec['v2'] = integration_limits[1]
    spec['ncol'] = np.log10(column)
    # Symmetrical errors:
    # spec['ncol_err_lo'] = -column_err_total/column*np.log10(np.e)
    # spec['ncol_err_hi'] = column_err_total/column*np.log10(np.e)
    # Asymmetrical errors:
    spec['ncol_err_lo'] = log_n_err_lo
    spec['ncol_err_hi'] = log_n_err_hi

    spec['flag_sat'] = flag_sat

    # Fill the Na(v) arrays
    spec['Nav'] = nav_array
    spec['Nav_err'] = nav_err_tot
    spec['Nav_sat'] = idx_saturation

    # Add the weights of the pixel integrations
    spec['integration_weights'] = weights

    if 'efnorm' in spec.keys():
        spec['fnorm_err'] = spec['efnorm']
        spec['fnorm_err_contin'] = spec['efnorm']*0.
        spec['fnorm_err_stat'] = spec['efnorm']

        del spec['efnorm']
        del spec['efnorm1']
        del spec['efnorm2']


    try:
        del spec['ncole1']
        del spec['ncole2']
        del spec['ncolez']
    except:
        pass

    return spec



def pynn_eqwidth(spec_in,integration_limits = None,
                partial_pixels = True):

    spec = spec_in.copy()

    # FIX NON-WRITEABLE ARRAYS due to discontiguous
    # memory in some readsav inputs
    if ~spec['vel'].flags.writeable:
        spec = fix_unwriteable_spec(spec)

    # Make sure there are integration limits:
    if integration_limits is None:
        integration_limits = [spec['v1'],spec['v2']]

    # Some constants and flags
    column_factor = 2.654e-15
    ew_factor = 1.13e17
    lightspeed = 2.998e5 # km/s

    velocity = spec['vel'].copy()
    flux = spec['flux'].copy()
    flux_err = spec['eflux'].copy()
    wavc=spec['wavc']
    fval=spec['fval']

    # Deal with the continuum
    if "contin" in spec.keys():
        continuum=spec['contin'].copy()
    else:
        try:
            continuum=spec['cont'].copy()
        except:
            continuum=spec['ycon'].copy()

    if "contin_err" in spec.keys():
        continuum_err = spec['contin_err'].copy()
    else:
        try:
            continuum_err = spec['econt'].copy()
        except:
            continuum_err = spec['ycon_sig'].copy()


    # Define the limits of the integration:
    if not integration_limits:
        integration_limits = [spec['v1'],spec['v2']]

    # Work out partial pixel weighting
    weights = integration_weights(velocity,integration_limits)
    # Uniform weighting if not partial pixel weighting
    if not partial_pixels:
        weights = np.zeros_like(velocity)
        xlim1, xlim2 = xlimit(velocity,integration_limits)
        weights[xlim1:xlim2+1] = 1.0

    # Create the wavelength array
    try:
        wave=spec['wave'].copy()
    except:
        wave = spec['wavc']*(velocity/lightspeed)+spec['wavc']
        spec['wave'] = wave

    # An array of delta wavelength:
    delw = wave[1:]-wave[:-1]
    delw = np.concatenate((delw,[delw[-1]]))

    # Calculate the equivalent width
    eqw_int = np.sum((1.-flux/continuum)*delw*weights)
    # Random flux errors
    eqw_stat_err = \
        np.sqrt(np.sum((flux_err/continuum*delw*weights)**2))
    # Continuum errors
    eqw_cont_err = \
        np.sum(continuum_err*(flux/continuum**2)*delw*weights)

    # Zero point error
    # TODO: Check this calculation
    z_eps = 0.01
    eqw_zero_err = z_eps*eqw_int

    # Combine errors
    eqw_err = np.sqrt(eqw_stat_err**2 \
        +eqw_cont_err**2 + eqw_zero_err**2)

    spec['v1'] = integration_limits[0]
    spec['v2'] = integration_limits[1]

    # Store the EW in milliAngstrom
    spec['EW'] = eqw_int*1000.
    spec['EW_err'] = eqw_err*1000.
    spec['EW_err_stat'] = eqw_stat_err*1000.
    spec['EW_err_cont'] = eqw_cont_err*1000.
    spec['EW_err_zero'] = eqw_zero_err*1000.

    # Add the cumulative EW
    spec['EW_cumulative'] = \
      np.cumsum((1.-flux/continuum)*delw*weights)*1000.

    # Calculate linear column density and error.
    linear_ncol = \
      ew_factor*spec['EW']/(spec['fval']*spec['wavc']**2)
    linear_ncol2sig = 2.0* \
      ew_factor*spec['EW_err']/(spec['fval']*spec['wavc']**2)
    linear_ncol3sig = 3.0* \
      ew_factor*spec['EW_err']/(spec['fval']*spec['wavc']**2)

    # Fill the output column densities
    spec['ncol_linearCoG'] = np.round(np.log10(linear_ncol),4)
    spec['ncol_linear2sig'] = \
        np.round(np.log10(linear_ncol2sig),4)
    spec['ncol_linear3sig'] = \
        np.round(np.log10(linear_ncol3sig),4)

    # Pixel weighting factors
    spec['integration_weights'] = weights

    # Is the line detected at 2, 3 sigma?
    if spec['EW'] >= 2.*spec['EW_err']:
        spec['detection_2sig'] = True
    else:
        spec['detection_2sig'] = False

    if spec['EW'] >= 3.*spec['EW_err']:
        spec['detection_3sig'] = True
    else:
        spec['detection_3sig'] = False


    # Delete the old versions of the EW quantities
    try:
        del spec['w']
        del spec['w_es']
        del spec['w_ec']
        del spec['w_et']
        del spec['w_ez']
        del spec['col2sig']
        del spec['col3sig']
    except:
        pass

    return spec
'''
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


    return species'''

'''
def getVel(wave,l0,zabs):
    wave_center=l0*(zabs + 1.)
    vel=const.c.cgs.value*1e-5*(wave**2-wave_center**2)/(wave**2+wave_center**2) #in km/s
    return vel

def findV(zAbs, zEm):
    c_light=299792.458
    v = ((((1.0+zEm)**2) - ((1.0+zAbs)**2))/(((1.0+zEm)**2) + ((1.0+zAbs)**2))) * c_light
    return v
'''

def getdict(spectrum,species,z):
    dictionary = {}
    obs_wave = spectrum[1].data['WAVE'][0]
    flux = spectrum[1].data['FLUX'][0]
    error = spectrum[1].data['ERROR'][0]
    
    for specie in species:
        for transition in species[specie]:

            vel = getVel(obs_wave,species[specie][transition][0],z)
            vel_sel = np.where((vel<=500) & (vel>=-500))

            dictionary['{}_{}'.format(specie,transition)] = {'vel':vel[vel_sel],'flux':flux[vel_sel],'eflux':error[vel_sel],
            'wavc':species[specie][transition][0],
           'fval':species[specie][transition][1],
            'contin':np.ones(len(vel[vel_sel])),
            'contin_err':np.zeros(len(vel[vel_sel]))}
            
    return dictionary



    
def getproperty(dictionary,integ):
    s1 = pynn_istat(dictionary,integration_limits = [integ[0],integ[1]],partial_pixels = True)
    
    s2 = pynn_column(dictionary,integration_limits = [integ[0],integ[1]],partial_pixels = True)
    
    s3 = pynn_eqwidth(dictionary,integration_limits = [integ[0],integ[1]],partial_pixels = True)
        
    return dict(dict(s1,**s2),**s3)

############################
'''def getVel2(wave,l0):
    vel=const.c.cgs.value*1e-5*(wave-l0)/(l0) #in km/s
    return vel
'''
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
'''def widths(i,j,ion,transition):
    return abs(max(spec[i:j][ion+transition+'_Vel'])-min(spec[i:j][ion+transition+'_Vel']))
vwidths=np.vectorize(widths)'''
###############################
'''def getEW(d,zabs):
    d = d.reset_index(drop=True)
    ew=np.trapz(1-d['FLUX'],x=d['WAVELENGTH'])/(1+zabs)
    d['Del-W']=d[['WAVELENGTH']].shift(-1)-d[['WAVELENGTH']]
    err=(sum((d['ERROR']*d['Del-W'])**2))**0.5/(1+zabs)
    return(ew,err)'''
##############################

# function to translate the observed wavelength to rest wavelength
def restwavl(w,z):
    x=w/(1.0+z)
    return (x)
#############################

#########################################################
'''def fivesigmadet(x,y):
    try:
        val=min(spec[x:y]['FLUX'][spec[x:y]['FLUX']>0])
        loc_min=np.where(spec['FLUX']==val)[0][0]
        if (1-val)/spec['ERROR'][loc_min] >= 5:
            return True
        else:
            return False
    except:
        return False'''
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
'''
def widths(i,j,ion,transition,spec):
    return abs(max(spec[i:j][ion+transition+'_Vel'])-min(spec[i:j][ion+transition+'_Vel']))
vwidths=np.vectorize(widths)   
'''

def filter_and_transform_dictionary(dictionary, threshold):
    """
    This function is used to remove any absorption systems that are beyond the emission redshift of the quasar.
    """
    filtered_dict = {key: [min(values), max(values)] for key, values in dictionary.items() if any(val <= threshold for val in values)}
    return filtered_dict


def remove_empty_filter_dictionary(my_dict):
    """
    This function removes keys with empty values
    """
    filtered_dict = {k: v for k, v in my_dict.items() if v}
    return filtered_dict

def getdictsep(wave,flux,error,species,z):
    """
    
    This function is meant to create a dictionary consisting of keys that can be ingested by PyNorm. 
    Args:
        wave (array of floats): The observed wavelength array
        flux (array of floats): The normalized flux array
        error (array of floats): The array of errors associated with the normalized flux
        species (dict) : The dictionary comprising of information on atomic line data
        z (float): The redshift of the absorption system whose measurements are of interest.
    Returns:
        dict: A dictionary that can be ingested by PyNorm for performing line measurements.
    
    """
    
    dictionary = {}
    obs_wave = wave
    flux = flux
    error = error
    
    for specie in species:
        for transition in species[specie]:

            vel = getVel(obs_wave,species[specie][transition][0],z)
            #print (vel)
            
            vel_sel = np.where((vel<=4000) & (vel>=-4000))

            dictionary['{}_{}'.format(specie,transition)] = {'z':z,'vel':vel[vel_sel],'flux':flux[vel_sel],'eflux':error[vel_sel],
            'wavc':species[specie][transition][0],
            'fval':species[specie][transition][1],
            'contin':np.ones(len(vel[vel_sel])),
            'contin_err':np.zeros(len(vel[vel_sel]))}
            
    return dictionary


def rebin_spectra(vel1, flux1, vel2, flux2, new_vel):
    """
    This function is meant to rebin a spectrum onto a common velocity axis.

    Args:
        vel1: velocity array of the first spectrum
        flux1: flux array of the first spectrum
        vel2: velocity array of the second spectrum
        flux2: flux array of the second spectrum        

    Returns:
        arrays: flux arrays of the first and second spectra.
    """
    
    interp_flux1 = interp1d(vel1, flux1, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_flux2 = interp1d(vel2, flux2, kind='linear', bounds_error=False, fill_value=np.nan)

    new_flux1 = interp_flux1(new_vel)
    new_flux2 = interp_flux2(new_vel)

    return new_flux1, new_flux2


def find_overlapping_bounds(outer_dict):
    """
    This function finds absorption components that overlap in redshift. 
    These overlapping components could be absorption arising from different HI transitions of the same absorption system.
    However, these components are vetted further below to remove false positives.
    """
    
    overlapping_outer_inner_keys = []

    outer_keys = list(outer_dict.keys())

    for i in range(len(outer_keys)):
        for j in range(i + 1, len(outer_keys)):
            outer_key1, outer_key2 = outer_keys[i], outer_keys[j]
            inner_dict1, inner_dict2 = outer_dict[outer_key1], outer_dict[outer_key2]

            for key1, bounds1 in inner_dict1.items():
                for key2, bounds2 in inner_dict2.items():
                    if (bounds1[0] <= bounds2[1] and bounds1[1] >= bounds2[0]) or \
                       (bounds2[0] <= bounds1[1] and bounds2[1] >= bounds1[0]):
                        overlapping_outer_inner_keys.append(((outer_key1, key1), (outer_key2, key2)))

    combined_tuples = []
    for i, tuple1 in enumerate(overlapping_outer_inner_keys):
        combined_tuple = list(tuple1)
        for j, tuple2 in enumerate(overlapping_outer_inner_keys[i + 1:]):
            if set(tuple1).intersection(set(tuple2)):
                combined_tuple.extend(x for x in tuple2 if x not in tuple1)
        combined_tuples.append(tuple(combined_tuple))

    return combined_tuples


def split_tuples(tuple_list):
    """
    Split a list of tuples into groups based on increasing first elements.

    This function takes a list of tuples and splits it into groups where the first
    element of each tuple is in ascending order within each group. A new group is
    started whenever the first element of a tuple is less than or equal to the
    first element of the previous tuple.

    Parameters:
    tuple_list (list): A list of tuples, where each tuple has at least one element.

    Returns:
    list: A list of tuples, where each tuple contains a group of original tuples
          with increasing first elements.
    
    """

    
    result = []
    current_group = []

    for tpl in tuple_list:
        if not current_group or tpl[0] > current_group[-1][0]:
            current_group.append(tpl)
        else:
            result.append(tuple(current_group))
            current_group = [tpl]

    if current_group:
        result.append(tuple(current_group))

    return result



def getinfozblock(wave,flux,err,block):
    """
    Analyze spectral data for a given block of transitions.

    This function processes spectral data for a set of transitions, calculates
    redshift ranges, and extracts properties for each transition.

    Parameters:
    wave (array-like): Wavelength data of the spectrum.
    flux (array-like): Flux data of the spectrum.
    err (array-like): Error data of the spectrum.
    block (list): A list of tuples, each containing element and transition information.

    Returns:
    dict: A dictionary containing properties for each unique transition in the block.
          Each key is a transition identifier, and the value is a dictionary of
          spectral properties for that transition.

    Notes:
    - The function uses several external dictionaries and functions (pr_dict_n,
      transition_library, myfuncpyNorm, etc.) which should be defined elsewhere
      in the code.
    - It calculates a common center of redshifts for the given transitions.
    - The function focuses on Hydrogen I (HI) transitions.
    - Properties are calculated within velocity ranges determined by the redshift
      limits of each transition.
    """
    
    z_element = []
    choose_transitions = []
    for element in block:
        z_element.append(pr_dict_n[element[0]][element[1]])
        choose_transitions.append(element[1])

    
    min_first = min(z_element, key=lambda x: x[0])[0]
    max_second = max(z_element, key=lambda x: x[1])[1]
    center = 0.5*(min_first+max_second)


    use_species = speciesinterest(['HI'],transition_library,choose={'HI':list(set(choose_transitions))})
    makeinpdict = getdictsep(wave,flux,err,use_species,center)


    z_l,z_h = [],[]
    for num, element in enumerate(block):    
        z_l.append((element[1],pr_dict_n[element[0]][element[1]][0])) 
        z_h.append((element[1],pr_dict_n[element[0]][element[1]][1]))
        

    unique_values_min = {}

    #print (z_l)
    for key, value in z_l:
        if key not in unique_values_min:
            unique_values_min[key] = value
        else:
            unique_values_min[key] = min(unique_values_min[key], value)
    
    # Creating a list of unique tuples
    unique_tuples_min = [(key, value) for key, value in unique_values_min.items()]
    #print (unique_tuples_min)

    unique_values_max = {}
    
    for key, value in z_h:
        if key not in unique_values_max:
            unique_values_max[key] = value
        else:
            unique_values_max[key] = max(unique_values_max[key], value)
    
    # Creating a list of unique tuples
    unique_tuples_max = [(key, value) for key, value in unique_values_max.items()]


    properties = {}

    for num,transition in enumerate(unique_tuples_min):
        v1,v2 = -findV(unique_tuples_min[num][1],center), findV(center,unique_tuples_max[num][1])

        vmin,vmax = min(makeinpdict['HI_{}'.format(transition[0])]['vel']),max(makeinpdict['HI_{}'.format(transition[0])]['vel'])

        
        v1 = v1 if vmin<v1 else vmin
        v2 = v2 if vmax>v2 else vmax

        
        
        
        properties[transition[0]] = getproperty(makeinpdict['HI_{}'.format(transition[0])],[v1,v2])
    
    return properties

def get_merged_transitions_tuples(inp):
    """
    Merge and group spectral transitions based on velocity proximity.

    This function analyzes multiple spectral transitions, identifies peaks,
    and groups transitions that have matching peaks within a 10 km/s velocity range.

    Parameters:
    inp (dict): A dictionary where keys are transition identifiers and values are
                dictionaries containing 'vel' (velocity), 'Nav' (flux), and 'z' (redshift) data.

    Returns:
    list of tuples: Each tuple contains:
                    - A velocity key (float, rounded to 2 decimal places)
                    - A tuple of transition information, where each item is a tuple of:
                      (velocity, transition identifier, calculated absorption redshift)

    Notes:
    - The function uses numpy for quantile calculations and scipy's find_peaks for peak detection.
    - Peaks are identified using a width of 5, and a height/prominence threshold of the 25th percentile
      of positive flux values.
    - Transitions are considered matching if their peak velocities are within 10 km/s of each other.
    - The returned list is sorted by the velocity key.
    - The function assumes the existence of a 'findZAbs' function to calculate absorption redshift.
    """
    
    merged_transitions = {}

    transitions_dict = inp
    for idx, (transition1, transition2) in enumerate(combinations(transitions_dict.keys(), 2)):
        velocity1 = transitions_dict[transition1]['vel']
        flux1 = transitions_dict[transition1]['Nav']
        velocity2 = transitions_dict[transition2]['vel']
        flux2 = transitions_dict[transition2]['Nav']
    
        z1 = transitions_dict[transition1]['z']
        z2 = transitions_dict[transition2]['z']
        
        use_height_1 = np.quantile(flux1[flux1>0],0.25)
        use_height_2 = np.quantile(flux2[flux2>0],0.25)
    
        flux1_ = flux1[flux1>use_height_1]
        flux2_ = flux2[flux2>use_height_2]
        
        
        # Find peaks in flux arrays
        peaks1, _ = find_peaks(flux1, width = 5, height = use_height_1,prominence = use_height_1)
        peaks2, _ = find_peaks(flux2, width = 5, height = use_height_2,prominence = use_height_2)
        

        
        # Check for matching peaks within 10 km/s
        for peak1 in peaks1:
            for peak2 in peaks2:
                if abs(velocity1[peak1] - velocity2[peak2]) <= 10:
                    # Group matching transitions based on velocity proximity
                    key = round(velocity1[peak1], 2)  # Using velocity as key (rounded to two decimal places)
                    if key not in merged_transitions:
                        merged_transitions[key] = set()  # Initialize set if key not present
                    merged_transitions[key].add((velocity1[peak1], transition1, findZAbs(-velocity1[peak1],z1)))
                    merged_transitions[key].add((velocity2[peak2], transition2, findZAbs(-velocity2[peak2],z2)))
    
    # Convert sets to tuples
    merged_transitions_tuples = [(key, tuple(value)) for key, value in merged_transitions.items()]
    merged_transitions_tuples.sort(key = lambda x: x[0])
    return merged_transitions_tuples



def cossim(t1,t2,makeinpdict):
    """
    Calculate the cosine similarity between two spectra.

    This function processes two spectra, normalizes them, and calculates their cosine similarity.
    It handles rebinning of spectra to ensure they are on the same velocity grid before comparison.

    Parameters:
    t1 (str or int): Identifier for the first spectrum in makeinpdict.
    t2 (str or int): Identifier for the second spectrum in makeinpdict.
    makeinpdict (dict): A dictionary containing spectral data for different identifiers.

    Returns:
    float: The cosine similarity between the two spectra, ranging from -1 to 1.
           1 indicates perfect similarity, 0 indicates no similarity, and -1 indicates perfect dissimilarity.

    Notes:
    - The function uses a velocity range of -100 to 100 km/s for the comparison.
    - Spectra are rebinned to a common velocity grid if necessary.
    - The spectra are normalized before calculating cosine similarity.
    - Uses external functions: myfuncpyNorm.getproperty, rebin_spectra, and cosine_similarity.
    """
    
    properties1 = getproperty(makeinpdict[t1],[-100,100])
    properties2 = getproperty(makeinpdict[t2],[-100,100])

    vel1 = properties1['vel']
    vel_sel1 = np.where((vel1<=100) & (vel1>=-100))
    vel1_use = vel1[vel_sel1]
    spectrum1 = properties1['Nav'][vel_sel1]
    
    vel2 = properties2['vel']
    vel_sel2 = np.where((vel2<=100) & (vel2>=-100))
    vel2_use = vel2[vel_sel2]
    spectrum2 = properties2['Nav'][vel_sel2]
    
    
    spectrum1,spectrum2 = rebin_spectra(vel1_use, spectrum1, vel2_use, spectrum2, np.arange(min(vel1_use[0],vel2_use[0]),max(vel1_use[-1],vel2_use[-1])+vel1[1]-vel1[0],vel1[1]-vel1[0]))
    
    spectrum1_norm = spectrum1 / np.linalg.norm(spectrum1)
    spectrum2_norm = spectrum2 / np.linalg.norm(spectrum2)
    
    # Reshape spectra into column vectors (required for cosine similarity calculation)
    spectrum1_norm = spectrum1_norm.reshape(-1, 1)
    spectrum2_norm = spectrum2_norm.reshape(-1, 1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(spectrum1_norm.T, spectrum2_norm.T)[0, 0]   

    return similarity



def unique_pairs(items):
    return list(combinations(items, 2))
def at_least_three_above_threshold(numbers, threshold=0.6):
    return sum(1 for num in numbers if num > threshold) >= 3

def redshiftgood(redshift):
    import warnings
    warnings.filterwarnings("ignore")
    cov_transitions=[]
    for transition_check in species['HI']:
        if  wave[0] <= species['HI'][transition_check][0]*(1.0+redshift) <= wave[-1]:
            cov_transitions.append(transition_check)
    use_species = speciesinterest(['HI'],transition_library,choose={'HI':cov_transitions})
    
    makeinpdict = getdictsep(wave,flux,err,use_species,redshift)

    ups = unique_pairs(cov_transitions)
    
    cossim_values = OrderedDict()
    for up in ups:
        cossim_values[up] = cossim('HI_{}'.format(up[0]),'HI_{}'.format(up[1]),makeinpdict) 
    
    firstquant, median, thirdquant = np.quantile(list(cossim_values.values()),[0.25,0.5,0.75])
    minval, maxval = np.min(list(cossim_values.values())),np.max(list(cossim_values.values()))


    if (maxval >= 0.8) and (at_least_three_above_threshold(list(cossim_values.values()))):
        return 1
    else:
        return 0
    

    