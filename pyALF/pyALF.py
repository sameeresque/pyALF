#!/usr/bin/python
import os
import sys
import pandas as pd
import asdf
import spectres
import pynorm
from pynorm.io import read_inorm
from pynorm.aod import pyn_batch
from scipy.interpolate import interp1d
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from scipy.signal import argrelextrema, argrelmax
from collections import OrderedDict
from functions_pyALF import *
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,FuncFormatter,
                               AutoMinorLocator)
import warnings
warnings.filterwarnings("ignore")

# read in the atom data file
transition_library=pd.read_table('atomdata_updated_new.dat', sep='\s+',header=None,comment = "#")
transition_library=np.asarray(transition_library)

#qso = 'J121930+494052'
qso = sys.argv[1]
# create folder for images
if not os.path.exists('./{0}_images'.format(qso)):
    os.makedirs('./{0}_images'.format(qso))


 
filein = '../Example/{0}.asdf'.format(qso)
plot_ions=['HI']
af = asdf.open(filein)
wave = np.asarray(af['wave'])
wave = np.array(wave, dtype="<f8")
flux = np.asarray(af['flux'])
flux = np.array(flux, dtype="<f8")
err = np.asarray(af['err'])
err = np.array(err, dtype="<f8")
d = {'WAVELENGTH':wave,'FLUX':flux,'ERROR':err}
spec=pd.DataFrame(data=d)
species=speciesinterest(plot_ions,transition_library,choose=None)
zem = af['zqso']
blue_limit, max_red = blue_red_limits(zem,spec['WAVELENGTH'],species,flag=1)
selected_spec = fluxselector(spec,blue_limit,max_red)

selected_spec['Rest-Wavelength']=selected_spec['WAVELENGTH']/(1.0+zem)
pr = absorptionlocator(selected_spec)
selected_spec = addvel2spec(selected_spec,species)



pr_dict=OrderedDict()
for num,pr_i in enumerate(pr):
    
    z_window = OrderedDict()
    wave_window = selected_spec[pr_i[0]:pr_i[1]]['WAVELENGTH']
    for transition in species['HI']:
        z_window[transition] = (wave_window/species['HI'][transition][0]) - 1.0
    filtered_dictionary = filter_and_transform_dictionary(z_window, zem)
    pr_dict[num] = filtered_dictionary   


pr_dict_n = remove_empty_filter_dictionary(pr_dict)

#t1 = find_overlapping_bounds(pr_dict_n)
#pickle.dump(t1,open('overlappingbounds_J121930+494052.pkl','wb'),protocol=2)


t1 = pd.read_pickle('../Example/overlappingbounds_{0}.pkl'.format(qso))


new_list = []
for num_t1,blk in enumerate(t1):
    result_list = split_tuples(blk)
    new_list.append(result_list)
import itertools
merged = list(itertools.chain(*new_list))


filtered_list = [tpl for tpl in merged if any(subtpl[1] == '1215' for subtpl in tpl)]


# time consuming steps#
# redshift_list = []
# for num,list_ in enumerate(filtered_list):
#     print (num)
#     inp = getinfozblock(wave,flux,err,list_)
#     output_list = get_merged_transitions_tuples(inp)
#     for item_ in output_list:
#         for vel in item_[1]:
#             redshift_list.append(vel[2])
#number_counts = Counter(redshift_list)
#result_list = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
#pickle.dump(result_list,open('resultlist_J121930+494052.pkl','wb'),protocol=2)

res_list = pd.read_pickle('../Example/resultlist_{0}.pkl'.format(qso))
#sorted_res_list =sorted(res_list, key=lambda element: element[0])


#pickle.dump(result_list,open('resultlist_J121930+494052.pkl','wb'),protocol=2)


# redshift_good = []
# for num, val in enumerate(res_list):
#     redshift_good.append(redshiftgood(val[0]))
#selected_res = [res for res, good in zip(res_list, redshift_good) if good == 1]
#pickle.dump(selected_res,open('selected_res_J121930+494052.pkl','wb'),protocol=2)
selected_res = pd.read_pickle('../Example/selected_res_{0}.pkl'.format(qso))

## plotting 



#plt.style.use('seaborn-white')
#plt.rc('font', family='serif')
fig = plt.figure(figsize=(9,14))

ax = fig.add_subplot(511)
ax1 = fig.add_subplot(512,sharex = ax)
ax2 = fig.add_subplot(513,sharex = ax)
ax3 = fig.add_subplot(514,sharex = ax)
ax4 = fig.add_subplot(515,sharex = ax)

for num,res in enumerate(selected_res[0:20]):
    plt.cla()
    ax.clear()
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    vel = getVel(wave,species['HI']['1215'][0],res[0])
    
    vel_sel = np.where((vel<=500) & (vel>=-500))
    ax.errorbar(vel[vel_sel],flux[vel_sel],color='gray', yerr=err[vel_sel],fmt='.',ls='none',label='HI-1215')
    
    vel = getVel(wave,species['HI']['1025'][0],res[0])
    
    vel_sel = np.where((vel<=500) & (vel>=-500))
    ax1.errorbar(vel[vel_sel],flux[vel_sel],color='gray', yerr=err[vel_sel],fmt='.',ls='none',label='HI-1025')
    
    vel = getVel(wave,species['HI']['972'][0],res[0])
    
    vel_sel = np.where((vel<=500) & (vel>=-500))
    ax2.errorbar(vel[vel_sel],flux[vel_sel],color='gray', yerr=err[vel_sel],fmt='.',ls='none',label='HI-972')
    
    vel = getVel(wave,species['HI']['949'][0],res[0])
    
    vel_sel = np.where((vel<=500) & (vel>=-500))
    ax3.errorbar(vel[vel_sel],flux[vel_sel],color='gray', yerr=err[vel_sel],fmt='.',ls='none',label='HI-949')
    
    vel = getVel(wave,species['HI']['937'][0],res[0])
    
    vel_sel = np.where((vel<=500) & (vel>=-500))
    ax4.errorbar(vel[vel_sel],flux[vel_sel],color='gray', yerr=err[vel_sel],fmt='.',ls='none',label='HI-937')
    
    for axi in [ax,ax1,ax2,ax3,ax4]:
        axi.legend(frameon=True,loc='best')
        axi.axhline(y=1,linestyle='--',color='black')
        axi.tick_params(axis='both', direction='in', which='major', length=4, width=1,labelsize=16)
        axi.tick_params(axis='both', direction='in', which='minor', length=2, width=1,labelsize=16)
        axi.yaxis.set_major_locator(MultipleLocator(0.5))
        axi.yaxis.set_minor_locator(MultipleLocator(0.1))
        axi.xaxis.set_major_locator(MultipleLocator(100))
        axi.xaxis.set_minor_locator(MultipleLocator(50))

    fig.text(0.5, 0.05, r'Relative Velocity [km s$^{-1}$]', ha='center', va='center',fontsize=20)
    fig.text(0.03, 0.5, 'Normalized Flux', ha='center', va='center', rotation='vertical',fontsize=20)

    ax.set_title('{}'.format(res[0]))
    plt.savefig('./{0}/{1}.png'.format(qso,num), bbox_inches='tight')
