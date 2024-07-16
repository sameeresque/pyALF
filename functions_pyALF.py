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


def getVel(wave,l0,zabs):
    wave_center=l0*(zabs + 1.)
    vel=const.c.cgs.value*1e-5*(wave**2-wave_center**2)/(wave**2+wave_center**2) #in km/s
    return vel

def findV(zAbs, zEm):
    c_light=299792.458
    v = ((((1.0+zEm)**2) - ((1.0+zAbs)**2))/(((1.0+zEm)**2) + ((1.0+zAbs)**2))) * c_light
    return v


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
