# pyALF: Installing `pyALF`

## Before installing pyALF, it is suggested to create a conda environment for pyALF
conda create -n pyALF python=3.10
pip install -r requirements.txt

## In addition to installing the packages in requirements.txt, it is recommended to install pyNorm which provides some useful functions for the measurement of absorption line properties 
```
pip install git+https://github.com/jchowk/pyNorm.git
export PYTHONPATH="$PYTHONPATH:/path/to/pyNorm/"
Rename the "io" folder to "ios" to avoid a conflict with some system files
```

## **Download from GitHub and install via `pip`:**

```
git clone https://github.com/sameeresque/pyALF/
cd pyALF
pip install .

(Note: pip will use setup.py to install your module. Avoid calling setup.py directly.)
```

## **Include `pyALF` in your `$PYTHONPATH`:**

Add the full path to the `pyALF` code to your `$PYTHONPATH` variable by invoking, or better yet add the path to .bashrc and .profile files

```
export PYTHONPATH="$PYTHONPATH:/path/to/pyALF/etc/"
```

## **To Uninstall pyALF**

```
pip uninstall pyALF
```
