# pyALF
A python Absorption Line Finding Tool. pyALF looks for weak HI absorption lines in the spectra of quasars, and returns a list of absorption line systems. This is done in two steps:

1. Search for 'genuine' absorbers.
2. Calculate the redshift of each probable Lyman alpha absorption, and look for corresponding Lyman series lines. By default, the code checks for all possible lines but an absorption system is flagged 'genuine' if atleast HI 1216 and HI 1026 are detected.

Here's a detailed <a href="https://github.com/sameeresque/pyALF/blob/main/docs/_build/html/index.html" title="Documentation">Documentation</a>.

# Installing `pyALF`

Before installing pyALF, it is suggested to create a conda environment for pyALF
```
conda create -n pyALF python=3.10
```
Activate the conda environment
```
conda activate pyALF
```
In addition to installing the packages in requirements.txt, you will need to install pyNorm which provides some useful functions for the measurement of absorption line properties 
```
pip install git+https://github.com/jchowk/pyNorm.git
```
## **Include `pyNorm` in your `$PYTHONPATH`:**

Add the full path to the `pyNorm` code to your `$PYTHONPATH` variable in your .bash_profile or .zshrc (newer MacOS) file:

```
export PYTHONPATH="$PYTHONPATH:/path/to/pyNorm/pyNorm/"
```

Note the path has to point to the subdirectory `pyNorm/pyNorm/`. 
To look up the location of pynorm:

```
pip show pyNorm
```

## **Download `pyALF` from GitHub and install via `pip`:**

```
git clone https://github.com/sameeresque/pyALF/
cd pyALF
pip install .
```

## **Editable `pip` installation:**

This installation approach allows edits to the code / `git pull` updates to be directly accessible. To enable this installation, invoke the following in the terminal from within the `pyALF` directory:
```
pip install -e .
```
Note: pip will use setup.py to install your module. Avoid calling setup.py directly.

## **Include `pyALF` in your `$PYTHONPATH`:**

Add the full path to the `pyALF` code to your `$PYTHONPATH` variable by invoking, or better yet add the path to .bashrc and .profile files

```
export PYTHONPATH="$PYTHONPATH:/path/to/pyALF/etc/"
```

# **To Uninstall pyALF**

```
pip uninstall pyALF
```

![openart-image_sw_Of4bB_1719800298641_raw](https://github.com/sameeresque/pyALF/assets/16863470/34cfa66d-bcb2-4582-b177-bca7de52c2ba)
