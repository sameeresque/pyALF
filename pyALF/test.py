from pipeline import pyALF
alf=pyALF('J121930+494052','../Example/J121930+494052.asdf','output')
alf.overlappingbounds()
alf.read_overlappingbounds()
alf.find_absorbers()
alf.read_absorbers()

# Traceback (most recent call last):
#  File "/Users/salonideepak/pyALF/pyALF/trial_run.py", line 5, in <module>
#    alf.find_absorbers()
#  File "/Users/salonideepak/pyALF/pyALF/pipeline.py", line 165, in find_absorbers
#    inp = getinfozblock(self.wave,self.flux,self.err,list_,self.pr_dict_n)
#  File "/Users/salonideepak/pyALF/pyALF/functions_pyALF.py", line 1096, in getinfozblock
#    use_species = speciesinterest(['HI'],transition_library,choose={'HI':list(set(choose_transitions))})
# NameError: name 'transition_library' is not defined