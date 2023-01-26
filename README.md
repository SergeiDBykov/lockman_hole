# Optical identification of X-ray sources in the SRG/eROSITA survey of Lockman Hole


These are scripts and notebooks which are used to identify optical counterparts for X-ray sources detected in the SRG/eROSITA Lockman Hole survey (Gilfanov et al. 2023)

The analysis results I used in my [paper](). Necessary python packages: `numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `astropy`, `sklearn`, `tensorflow` (`keras`), `tqdm`, [`nway`](https://github.com/SergeiDBykov/nway).

Note that the data (e.g. from Chandra, XMM, DESI) is not included and needed to be downloaded separately. eROSITA data from Lockman Hole will be available in the future. All data should be placed in `./0_data` folder

## The structure of the code is as follows:

- `./scripts`  
  the main scripts for the analysis are placed here. It containt a few files to manage the cross-match problem.
    * `./utils.py` is used to set up pathes and contains some utility/plotting functions
    * `./cross_match_scripts.py` contains functions for managing the catalogs, building machine learning models,  and calculating the needed survey metrics
    * `./viewer.py` contains functions for visualising optical fields around X-ray sources.


<br>

- `./notebooks`  
  the main notebooks for the analysis are stored here. It contains a few folders and separate notebooks to make the identification and its validation. The explanations are given in each notebook.  The content of the folder is as follows (roughly in the order of the paper sections):
    * `./1_desi-photo-prior` contains notebooks to build photometric priors with machine learning models.
        * `0_train-catalogs.ipynb`  creates a training data for prior model from Chandra and DESI LIS catalogs.
        * `1_train-catalogs-preprocessing` contains notebooks to preprocess the training data.
        * `2_prior-learning-keras-nnmag.ipynb` trains the prior models with `keras` and `scikit-learn`, and saves the models.
        * `3_prior-learning-fain-features.ipynb` trains the prior based only on the distibution of the features (for faint sources).
    
    * `./2_desi-validation-catalog` contains notebooks to cross-match the X-ray (eROSITA, Chandra, XMM) and optical (DESI LIS) catalogs to create a validation sample in the Lockman Hole area.
        * `0_catalog-preparation-ero-csc-xmm.ipynb` prepares the catalogs from eROSITA, Chandra, and XMM.
        * `1_validation-counterparts.ipynb` creates a sample of robust indentifications (true counterparts).
        * `2_validation-hostless.ipynb` creates a sample of robust hostless X-ray sources (thue hostless).
    * `./3_desi-crossmatch` contains notebooks for cross-match with the priors learned.
        * `0.0_catalog-preparation-erosita.ipynb` prepares eROSITA data for cross-match with NWAY code.
        * `0.1_catalog-preparation-desi.ipynb` prepares DESI LIS data for cross-match with NWAY code. Adds photometric priors according to the learned models and available data. 
        * `1_magnn-match.ipynb` cross-matches the eROSITA and DESI LIS catalogs with NWAY code. It uses slightly modified version of the code, which is available [here](https://github.com/SergeiDBykov/nway).
        * `2_distance-match.ipynb` the same as above but without photometric priors.
        * `3.1_results.ipynb` presents the cross-match results and metrics. 
        * `3.2_viewer.ipynb` shows example of optical fields around X-ray sources for correct and incorrect identifications.


