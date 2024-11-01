#imports#
import os
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

# read in the atom data file
transition_library=pd.read_table('atomdata_updated_new.dat', sep='\s+',header=None,comment = "#")
transition_library=np.asarray(transition_library)

qso = 'J121930+494052'
filein = 'Example/{0}.asdf'.format(qso)
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


t1 = pd.read_pickle('Example/overlappingbounds_J121930+494052.pkl')



new_list = []
for num_t1,blk in enumerate(t1):
    result_list = split_tuples(blk)
    new_list.append(result_list)
import itertools
merged = list(itertools.chain(*new_list))


filtered_list = [tpl for tpl in merged if any(subtpl[1] == '1215' for subtpl in tpl)]

def redshift(trans):
    """
    This function is used to return the redshift of an absorption system given a list of overlapping HI.
    """
    #print(trans)
    inp = getinfozblock(wave,flux,err,trans,pr_dict_n)
    output_list = get_merged_transitions_tuples(inp)
    for item_ in output_list:
        for vel in item_[1]:
            redshift_list.append(vel[2])
    return redshift_list


if __name__ == '__main__':
    pool = Pool(processes=4)
    results = pool.map(redshift,filtered_list)
    print (results)