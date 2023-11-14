## BADFit

Code is conditional on the successful installation of Sherpa with access to XSpec models.

XSpec Installation: 
https://heasarc.gsfc.nasa.gov/xanadu/xspec/

Sherpa Installation:
https://sherpa.readthedocs.io/en/latest/install.html

Source install with conda:
https://sherpa.readthedocs.io/en/latest/developer/index.html#source-install-with-conda

Refer to __[Tutorial.ipynb](https://github.com/samlaihei/BADFit/blob/main/Tutorial.ipynb)__ for a detailed guide.

The key features of this code are:
-  Provide a Bayesian framework to constrain BH properties from a provided Spectral Energy Distribution (SED) utilising ray-traced models of the multi-temperature thermal accretion disc emission around Kerr BHs. 
-  Use general extinction curve models to deredden the SED.
-  Define custom priors and parameter thresholds to influence the resulting model. Freely fix and vary different parameters.

## Acknowledgements

We thank Marcin Marculewicz, Ashley Hai Tung Tan, and Yanina Lopez Bonilla for testing the installation and raising several issues.

## Cite this code

The preferred citations for this code are the following:

> @ARTICLE{2023MNRAS.521.3682L,\
>       author = {{Lai (赖民希)}, Samuel and {Wolf}, Christian and {Onken}, Christopher A. and {Bian (边福彦)}, Fuyan},\
>        title = "{Characterising SMSS J2157-3602, the most luminous known quasar, with accretion disc models}",\
>      journal = {\mnras},\
>     keywords = {galaxies: active, galaxies: high-redshift, quasars: emission lines, Astrophysics - Astrophysics of Galaxies},\
>         year = 2023,\
>        month = may,\
>       volume = {521},\
>       number = {3},\
>        pages = {3682-3698},\
>          doi = {10.1093/mnras/stad651},\
>      archivePrefix = {arXiv},\
>       eprint = {2302.10397},\
>      primaryClass = {astro-ph.GA},\
>       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.3682L} ,\
>      adsnote = {Provided by the SAO/NASA Astrophysics Data System}\
> }


> @software{samuel_lai_2023_7772748,\
> author       = {Samuel Lai},\
> title        = {samlaihei/BADFit: BADFit v1.0.0},\
> month        = mar,\
> year         = 2023,\
> publisher    = {Zenodo},\
> version      = {v1.0.0},\
> doi          = {10.5281/zenodo.7772748},\
> url          = {https://doi.org/10.5281/zenodo.7772748} \
> }