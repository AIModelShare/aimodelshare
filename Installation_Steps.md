## How to Install AIModelshare 


### Conda/Mamba Install ( For MAC and Linux Users Only , Windows Users should use pip method ) : 

Make sure you have conda version >=4.9 

You can check your conda version with:

```
conda --version
```

To update conda use: 

```
conda update conda 
```

Installing `aimodelshare` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once the `conda-forge` channel has been enabled, `aimodelshare` can be installed with `conda`:

```
conda install aimodelshare
```

or with `mamba`:

```
mamba install aimodelshare
```


### PIP Installation ( For all Platforms) : 

Install `aimodelshare` using 

```
pip install aimodelshare
```
