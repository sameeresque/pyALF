#!/usr/bin/python
import os
import sys
import pandas as pd
import asdf
import spectres
import pynorm
from pynorm.ios import read_inorm
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


path = os.path.dirname(__file__)
#path = os.path.abspath(__file__).resolve().parent

class pyALF(object):
    '''
    The main function for the pyALF pipeline. This class is meant to be used as an end-to-end pipeline.
    The qso name (str), qso_file (asdf format) and output folder (str) are required to run the pipeline.
    '''    
    def __init__(self,qso, qso_file,output_folder):
        # read in the atom data file
        self.transition_library=np.asarray(pd.read_table(path+'/atomdata_updated_new.dat', sep='\s+',header=None,comment = "#"))
        self.filein = qso_file
        self.qso = qso
        self.output_folder = output_folder
        # create folder for images
        if not os.path.exists('{0}/{1}_images'.format(self.output_folder,self.qso)):
            os.makedirs('{0}/{1}_images'.format(self.output_folder,self.qso))
        self.plot_ions=['HI']
        self.af = asdf.open(self.filein)
        self.wave = np.asarray(self.af['wave'])
        self.wave = np.array(self.wave, dtype="<f8")
        self.flux = np.asarray(self.af['flux'])
        self.flux = np.array(self.flux, dtype="<f8")
        self.err = np.asarray(self.af['err'])
        self.err = np.array(self.err, dtype="<f8")
        self.d = {'WAVELENGTH':self.wave,'FLUX':self.flux,'ERROR':self.err}
        self.spec=pd.DataFrame(data=self.d)
        self.species=speciesinterest(self.plot_ions,self.transition_library,choose=None)
        self.zem = self.af['zqso']
        self.blue_limit, self.max_red = blue_red_limits(self.zem,self.spec['WAVELENGTH'],self.species,flag=1)
        self.selected_spec = fluxselector(self.spec,self.blue_limit,self.max_red)
        self.selected_spec['Rest-Wavelength']=self.selected_spec['WAVELENGTH']/(1.0+self.zem)
        self.pr = absorptionlocator(self.selected_spec)
        self.selected_spec = addvel2spec(self.selected_spec,self.species)

    def overlappingbounds(self):
        '''Search for the overlapping bounds in the selected spectrum and return an array.'''
        self.t1 = pd.read_pickle('/Users/dazhi/Downloads/codeastro/pyALF/Example/overlappingbounds_J121930+494052.pkl')

    def resultlist(self):
        '''(some processing and) Return the list of redshifts with their counts.'''
        self.res_list = pd.read_pickle('/Users/dazhi/Downloads/codeastro/pyALF/Example/resultlist_J121930+494052.pkl')

    def selected_res(self): 
        '''(some processing and) Return the selected redshifts.'''
        self.selected_res = pd.read_pickle('/Users/dazhi/Downloads/codeastro/pyALF/Example/selected_res_J121930+494052.pkl')

    def pipeline(self):
            '''
            The main pyALF pipeline function to provide the all possible HI absorbers with their line profiles.

            This is an end-to-end function for the pyALF package, it will read the quasar spectrum
            and make the plot for the possible HI absorbers with their profile in different transitions.
            
            Parameters:
            qso_file (str): The path to the quasar spectrum file in asdf format.
            output_folder (str): The path to the output folder where the plots will be saved.

            Note: 
            This function won't return anything, it will save the plots in the folder named as '{qso}_images'.
            '''


            pr_dict=OrderedDict()
            for num,pr_i in enumerate(self.pr):
                
                z_window = OrderedDict()
                wave_window = self.selected_spec[pr_i[0]:pr_i[1]]['WAVELENGTH']
                for transition in self.species['HI']:
                    z_window[transition] = (wave_window/self.species['HI'][transition][0]) - 1.0
                filtered_dictionary = filter_and_transform_dictionary(z_window, self.zem)
                pr_dict[num] = filtered_dictionary   


            pr_dict_n = remove_empty_filter_dictionary(pr_dict)

            #t1 = find_overlapping_bounds(pr_dict_n)
            #pickle.dump(t1,open('overlappingbounds_J121930+494052.pkl','wb'),protocol=2)

            new_list = []
            for num_t1,blk in enumerate(self.t1):
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

            #sorted_res_list =sorted(res_list, key=lambda element: element[0])


            #pickle.dump(result_list,open('resultlist_J121930+494052.pkl','wb'),protocol=2)


            # redshift_good = []
            # for num, val in enumerate(res_list):
            #     redshift_good.append(redshiftgood(val[0]))
            #selected_res = [res for res, good in zip(res_list, redshift_good) if good == 1]
            #pickle.dump(selected_res,open('selected_res_J121930+494052.pkl','wb'),protocol=2)

            ## plotting 



            #plt.style.use('seaborn-white')
            #plt.rc('font', family='serif')
            fig = plt.figure(figsize=(9,14))

            ax = fig.add_subplot(511)
            ax1 = fig.add_subplot(512,sharex = ax)
            ax2 = fig.add_subplot(513,sharex = ax)
            ax3 = fig.add_subplot(514,sharex = ax)
            ax4 = fig.add_subplot(515,sharex = ax)

            for num,res in enumerate(self.selected_res[0:20]):
                plt.cla()
                ax.clear()
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                vel = getVel(self.wave,self.species['HI']['1215'][0],res[0])
                
                vel_sel = np.where((vel<=500) & (vel>=-500))
                ax.errorbar(vel[vel_sel],self.flux[vel_sel],color='gray', yerr=self.err[vel_sel],fmt='.',ls='none',label='HI-1215')
                
                vel = getVel(self.wave,self.species['HI']['1025'][0],res[0])
                
                vel_sel = np.where((vel<=500) & (vel>=-500))
                ax1.errorbar(vel[vel_sel],self.flux[vel_sel],color='gray', yerr=self.err[vel_sel],fmt='.',ls='none',label='HI-1025')
                
                vel = getVel(self.wave,self.species['HI']['972'][0],res[0])
                
                vel_sel = np.where((vel<=500) & (vel>=-500))
                ax2.errorbar(vel[vel_sel],self.flux[vel_sel],color='gray', yerr=self.err[vel_sel],fmt='.',ls='none',label='HI-972')
                
                vel = getVel(self.wave,self.species['HI']['949'][0],res[0])
                
                vel_sel = np.where((vel<=500) & (vel>=-500))
                ax3.errorbar(vel[vel_sel],self.flux[vel_sel],color='gray', yerr=self.err[vel_sel],fmt='.',ls='none',label='HI-949')
                
                vel = getVel(self.wave,self.species['HI']['937'][0],res[0])
                
                vel_sel = np.where((vel<=500) & (vel>=-500))
                ax4.errorbar(vel[vel_sel],self.flux[vel_sel],color='gray', yerr=self.err[vel_sel],fmt='.',ls='none',label='HI-937')
                
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
                plt.savefig('{0}/{1}_images/{2}.png'.format(self.output_folder,self.qso,num), bbox_inches='tight')
