from .utils import data_path, set_mpl
from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.table import Table 
from astropy.coordinates import SkyCoord
from astropy import coordinates 
import astropy.io.fits as fits
import astropy.units as u
import healpy as hp
from scipy import stats
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
import tensorflow as tf
pd.options.mode.chained_assignment = None
from typing import Tuple, List, Optional, Dict, Callable

set_mpl()


def cat2hpx(lon: np.ndarray, lat: np.ndarray, nside: int, radec: bool = True) -> np.ndarray:
    """
    https://stackoverflow.com/questions/50483279/make-a-2d-histogram-with-healpix-pixellization-using-healpy
    Convert a catalogue to a HEALPix map of number counts per resolution
    element.
    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
        coordinate system. Otherwise assume input is glon (l), glat (b)
    nside : int
        HEALPix nside of the target map
    radec : bool
        Switch between R.A./Dec and glon/glat as input coordinate system.
    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates
    """

    npix = hp.nside2npix(nside)
    map_resol_deg = np.rad2deg(hp.nside2resol(nside))
    print(f'Resolution of the HEALPix map:')
    print(f'{map_resol_deg} deg per pixel, or')
    print(f'{map_resol_deg*60} arcmin per pixel, or')
    print(f'{map_resol_deg*60*60} arcsec per pixel')

    if radec:
        eq = SkyCoord(lon, lat, frame="icrs", unit="deg")
        l, b = eq.galactic.l.value, eq.galactic.b.value  # type: ignore
    else:
        l, b = lon, lat

    l, b = np.array(l), np.array(b)
    # conver to theta, phi
    theta = np.radians(90.0 - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)
    # unique because in a pixel there might be lots of sources
    idx, counts = np.unique(indices, return_counts=True)  # type: ignore

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts

    return hpx_map

def _add_at(a,index,b):
    np.add.at(a,index,b)    


def make_healpix_map(ra, dec, quantity, nside, mask=None, weight=None, ipix=None, fill_UNSEEN=False, return_w_maps=False, return_extra=False, mode='mean'):
    """
    source: https://github.com/xuod/castor/blob/master/castor/cosmo.py
    Creates healpix maps of quantity observed at ra, dec (in degrees) by taking
    the mean or sum of quantity in each pixel.
    Parameters
    ----------
    ra : array
        Right ascension.
    dec : array
        Declination.
    quantity : array
        `quantity` can be 2D, in which case several maps are created.
    nside : int
        `nside` parameter for healpix.
    mask : array
        If None, the mask is created and has value 1 in pixels that contain at
        least one object, 0 elsewhere.
    weight : type
        Weights of objects (the default is None, in which case all objects have
        weight 1). Must be the same size as `quantity`.
    ipix : array
        `ipix` should be the array of healpix pixel indices corresponding to the
        input `ra` and `dec`. By default it is None and will be computed.
    fill_UNSEEN : boolean
        If `fill_UNSEEN` is True, pixels outside the mask are filled with
        hp.UNSEEN, 0 otherwise (the default is False).
    return_extra : boolean
        If True, a dictionnary is returned that contains count statistics and
        the masked `ipix` array to allow for statistics on the quantities to be
        computed.
    mode : string
        Whether to return the 'mean' or 'sum' of quantity in each pixel.
    Returns
    -------
    List of outmaps, the count map and the mask map.
    """
    npix = hp.nside2npix(nside)
    map_resol_deg = np.rad2deg(hp.nside2resol(nside))
    print(f'Resolution of the HEALPix map:')
    print(f'{map_resol_deg} deg per pixel, or')
    print(f'{map_resol_deg*60} arcmin per pixel, or')
    print(f'{map_resol_deg*60*60} arcsec per pixel')
    if quantity is not None:
        quantity = np.atleast_2d(quantity)

        if weight is not None:
            w = np.atleast_2d(weight)
            # Weights can also be the same for all quantities
            # assert quantity.shape==weight.shape, "[make_healpix_map] quantity and weight must have the same shape"
            if w.shape[0] > 1:
                assert quantity.shape == w.shape, "[make_healpix_map] quantity/weight arrays don't have the same length"
            else:
                w = np.tile(w[0], (quantity.shape[0],1))

            assert np.all(w > 0.), "[make_healpix_map] weight is not strictly positive"
        else:
            w = np.ones_like(quantity)

        assert quantity.shape == w.shape, "[make_healpix_map] quantity/weight arrays don't have the same length"

    npix = hp.nside2npix(nside)

    if mask is not None:
        assert len(mask)==npix, "[make_healpix_map] mask array does not have the right length"

    # Value to fill outside the mask
    x = hp.UNSEEN if fill_UNSEEN else 0.0

    # Make sure mode is correct
    assert (mode in ['sum','mean']), "[make_healpix_map] mode should be 'mean' or 'sum'"

    count = np.zeros(npix, dtype=float)
    outmaps = []
    sum_w_maps = []

    # Getting pixels for each object
    if ipix is None:
        assert len(ra) == len(dec), "[make_healpix_map] ra/dec arrays don't have the same length"
        if quantity is not None:
            assert len(ra) == quantity.shape[1]
        ipix = hp.ang2pix(nside, (90.0-dec)/180.0*np.pi, ra/180.0*np.pi)
    else:
        if quantity is not None:
            assert len(ipix) == quantity.shape[1], "[make_healpix_map] ipix has wrong size"

    # Counting objects in pixels
    np.add.at(count, ipix, 1.)
    #_add_at_cst(count, ipix, 1.)

    # Creating the mask if it does not exist
    if mask is None:
        bool_mask = (count > 0)
    else:
        bool_mask = mask.astype(bool)

    # # Masking the count in the masked area
    # count[np.logical_not(bool_mask)] = x
    # if mask is None:
    #     assert np.all(count[bool_mask] > 0), "[make_healpix_map] count[bool_mask] is not positive on the provided mask !"

    # Create the maps
    if quantity is not None:
        for i in range(quantity.shape[0]):
            sum_w = np.zeros(npix, dtype=float)
            # np.add.at(sum_w, ipix, w[i,:])
            _add_at(sum_w, ipix, w[i,:])

            outmap = np.zeros(npix, dtype=float)
            # np.add.at(outmap, ipix, quantity[i,:]*w[i,:])
            _add_at(outmap, ipix, quantity[i,:]*w[i,:])

            if mode=='mean':
                outmap[bool_mask] /= sum_w[bool_mask]
                
            outmap[np.logical_not(bool_mask)] = x

            outmaps.append(outmap)
            if return_w_maps:
                sum_w_maps.append(sum_w)

    if mask is None:
        returned_mask = bool_mask.astype(float)
    else:
        returned_mask = mask

    res = [outmaps, count, returned_mask]

    if return_w_maps:
        res += [sum_w_maps]

    if return_extra:
        extra = {}
        extra['count_tot_in_mask'] = np.sum(count[bool_mask])
        extra['count_per_pixel_in_mask'] = extra['count_tot_in_mask'] * 1. / np.sum(bool_mask.astype(int))
        extra['count_per_steradian_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=False)
        extra['count_per_sqdegree_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=True)
        extra['count_per_sqarcmin_in_mask'] = extra['count_per_sqdegree_in_mask'] / 60.**2
        extra['ipix_masked'] = np.ma.array(ipix, mask=bool_mask[ipix])

        res += [extra]

    return res
#


def decode_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    decode_str_columns decodes all string columns in a pandas dataframe

    Args:
        df (pd.DataFrame): dataframe to decode

    Returns:
        pd.DataFrame: decoded dataframe

    """

    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df
    
def pandas_to_fits(dataframe: pd.DataFrame,
                    filename: str,
                    table_header_name: str,
                    sky_area_deg2: float):
    """
    pandas_to_fits saves a pandas dataframe as a fits file with all columns. Saves to data_path + filename.fits
    ##https://github.com/JohannesBuchner/srgz/blob/master/srgz-write-header.py
    Args:
        dataframe (pd.DataFrame): dataframe to save
        filename (str): filename (with path and .fits extension)
        table_header_name (str): header of the table, e.g. eROSITA-LHC
        sky_area_deg2 (float): sky area of the survey (needed for srgz)
    """
    table = Table.from_pandas(dataframe)
    table.write(data_path+'/'+filename, overwrite = True)

    #https://github.com/JohannesBuchner/srgz/blob/master/srgz-write-header.py
    with fits.open(data_path+'/'+filename , 'update') as file:
        file[1].name = table_header_name
        file[1].header['SKYAREA'] = sky_area_deg2
        file.flush()
    return None


def fits_to_pandas(filename: str, include_data_path: bool = True) -> pd.DataFrame:
    """
    fits_to_pandas reads a fits file and returns a pandas dataframe

    Args:
        filename (str): filename (with .fits extension)
        include_data_path (bool, optional): if True, the data_path is added to the filename. Defaults to True.

    Returns:
        pd.DataFrame: pandas dataframe
    """

    if include_data_path:
        filename = data_path+'/'+filename
    else:
        filename = filename
    data = Table.read(filename, format='fits')
    with fits.open(filename , 'readonly') as file:
        dataname = file[1].name
    #the next is to handle multi-dimensional columns
    names = [name for name in data.colnames if len(data[name].shape) <= 1]
    dataframe = data[names].to_pandas()
    dataframe.reset_index(inplace=True)
    dataframe.rename(columns={'index': dataname}, inplace=True)


    #convert bytes to strings
    str_df = dataframe.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        dataframe[col] = str_df[col]
    
    return dataframe


def my_scaler_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    my_scaler_forward: scales the columns of a dataframe according to the following rules:
        all magnitudes are scaled by 1/35
        all colors are scaled by 1/10
        all other columns are not scaled
    Args:
        df (pd.DataFrame): dataframe to scale

    Returns:
        pd.DataFrame: scaled dataframe
    """
    df_scaled = df.copy()
    for colname in df.columns:
        if 'rel_dered_mag' in colname:
            print(colname, 'scaled by 1/35')
            df_scaled[colname] = df[colname]/35
        elif 'rel_dered' in colname and 'mag_' not in colname:
            print(colname, 'scaled by 1/10')
            df_scaled[colname] = df[colname]/10
        else:
            df_scaled[colname] = df[colname]
    return df_scaled

def my_scaler_backward(df_scaled: pd.DataFrame)->pd.DataFrame:
    """
    my_scaler_backward unscales the columns of a dataframe which was scaled according to the following rules:
        all magnitudes are scaled by 1/35
        all colors are scaled by 1/10
        all other columns are not scaled
    Args:
        df_scaled (pd.DataFrame): dataframe to unscale

    Returns:
        pd.DataFrame: unscaled dataframe
    """
    df = df_scaled.copy()
    for colname in df.columns:
        if 'rel_dered_mag' in colname:
            df[colname] = df_scaled[colname]*35
        elif 'rel_dered' in colname and 'mag_' not in colname:
            df[colname] = df_scaled[colname]*10
        else:
            df[colname] = df_scaled[colname]
    print('data unscaled: rel_dered_mag for mags,  rel_dered and not mag_ for colors')
    return df


def assess_classifier(clf, X_test: np.ndarray, y_test: np.ndarray, label: str = 'Validation set', histbins: int = 30) -> Tuple:  
    """
    assess_classifier for a given X_test and y_test, plots the purity/completeness (precision/recall) curves and the histogram of the predicted probabilities/

    Args:
        clf (keras or sklearn model): model of the classifier
        X_test (np.ndarray): test features
        y_test (np.ndarray): test labels
        label (str, optional): label of the test dataset. Defaults to 'Validation set'.
        histbins (int, optional): number of bins. Defaults to 30.

    Returns:
        Tuple: optimal decision threshcold (precision=recall), precision and recall on this threshold, histogram of the predicted probabilities as per NWAY format
    """

    try:
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
    except:
        pass


    plt.figure(figsize=(5,5))
    try:
        predict_proba = clf.predict_proba(X_test)[:,1]
    except:
        #add predict_proba for consistency with sklearn
        clf.predict_proba = lambda X: np.vstack((np.ones(X.shape[0]), clf.predict(X, verbose = 0)[:,0])).T
        predict_proba = clf.predict_proba(X_test)[:,1]


    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, predict_proba)
    precision = precision[:-1]
    recall = recall[:-1]
    #recall is TP / (TP + FN) - We know it as  completeness  -> number of true positives / all positives
    #precision is TP / (TP + FP) - We know it as purity -> number of samples that are correctly classified out of all classified samples



    fig,  ax =  plt.subplots( figsize = (9,5))
    ax.plot(thresholds, precision , label=label+':putiry', linewidth=2)
    ax.plot(thresholds,recall , label=label+':completeness', linewidth=2)
    ax.set_xlabel('classifier output')
    ax.set_ylabel('completeness/purity')
    plt.grid(True)
    id_optim = np.argmin(np.abs(precision-recall))
    threshold_optim = thresholds[id_optim]
    precision_optim = precision[id_optim]
    recall_optim = recall[id_optim]
    print('Optimal threshold: {:.2f}'.format(threshold_optim))
    print('Optimal precision: {:.2f}'.format(precision_optim))
    ax.axvline(threshold_optim, color='C5', label = f"({precision_optim:.2f},{recall_optim:.2f})")

    cm = sklearn.metrics.confusion_matrix(y_test, predict_proba > threshold_optim)

    cm_str = f"TN: {cm[0,0]}  FP: {cm[0,1]}\nFN: {cm[1,0]}  TP: {cm[1,1]}"
    #add a string version of the confusion matrix, add true positive , false positive  etc labels

    ax.text(0.3, 0.15, cm_str, ha='center', va='center', transform=ax.transAxes)
    ax.legend(loc = 'lower right')

    y_test = np.reshape(y_test, (-1,))
    bins = np.linspace(0, 1, histbins)
    hist_field, bin_field = np.histogram(predict_proba[y_test==0], bins=bins, density=True)  
    hist_ctsp, bin_ctsp = np.histogram(predict_proba[y_test==1], bins=bins, density=True)

    plt.figure(figsize=(10,10))
    plt.bar(bin_field[:-1], hist_field, width=0.01, color='r', label='field sources', alpha = 0.5)
    plt.bar(bin_ctsp[:-1], hist_ctsp, width=0.01, color='b', label='counterpart sources', alpha = 0.5)
    plt.legend()
    plt.xlabel('classifier predicted probability')
    plt.ylabel('probability density')

    #with columns lo, hi, selected, others
    hist_df = pd.DataFrame({'lo':bin_field[:-1], 'hi':bin_field[1:], 'selected':hist_ctsp, 'others':hist_field})


    return threshold_optim, precision_optim, recall_optim, hist_df






def plot_metrics(history, metrics: list = ['loss', 'purity', 'completeness']):
    """
    plot_metrics plots the metrics of the training history (keras or sklearn)

    Args:
        history: keras history
        metrics (list, optional): metrics to plot. Defaults to ['loss', 'purity', 'completeness'].
    """
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.figure(figsize=(8,5))
        try:
            plt.plot(history.epoch, history.history[metric], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Test')
        except:
            plt.plot(history['epoch'], history[metric], label='Train')
            plt.plot(history['epoch'], history['val_'+metric], linestyle="--", label='Test')
        plt.xlabel('Epoch')
        plt.ylabel(name)

        plt.legend();



def build_keras_model(input_features_shape: Tuple,
                        activation: str='relu', 
                        layers_num: Tuple = (8,8,8),
                        dropout_rate: float = 0.0,
                        initial_bias: Optional[bool] = None,
                        lr: float = 1e-3,
                        load_weights: bool = True,) -> Tuple:
    """
    build_keras_model builds a keras model

    Args:
        input_features_shape (Tuple): input features shape
        activation (str, optional): activation. Defaults to 'relu'.
        layers_num (Tuple, optional): number of nodes in layers. Defaults to (8,8,8).
        dropout_rate (float, optional): dropout rate. Defaults to 0.0.
        initial_bias (Optional[bool], optional): inintial bias. Defaults to None.
        lr (float, optional): learning rate. Defaults to 1e-3.
        load_weights (bool, optional): whether to load weights. Defaults to True.

    Returns:
        Tuple: keras model and early stopping callback
    """
    

    #initial_bias = np.log([np.sum(y_test)/np.sum(~y_test)])
    #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    METRICS = [
        keras.metrics.Precision(name='completeness'),
        keras.metrics.Recall(name='purity'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]


    #recall is TP / (TP + FN) - We know it as purity
    #precision is TP / (TP + FP) - We know it as completeness

    def make_model(metrics=METRICS, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        layers = [keras.layers.Dense(layers_num[0], activation=activation,
        input_shape=(input_features_shape,))]

        for num in layers_num[1:]:
            layers.append(keras.layers.Dense(num, activation=activation))
            layers.append(keras.layers.Dropout(dropout_rate))

        layers += [keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),]

        model = keras.Sequential(layers)


        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)
        return model


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        verbose=1,
        patience=20,
        mode='auto',
        restore_best_weights=load_weights)


    model = make_model(output_bias=initial_bias)
    model.summary()
    return model, early_stopping



def photo_prior_create_train_test_validation_data(photo_cat_scaled: pd.DataFrame, 
                                                x_ray_flux_bins_num: int = 1,
                                                features_cols: str = 'grzw1w2',
                                                validation_fraction: float = 0.3,
                                                test_fraction: float = 0.2,
                                                downsample_field_srcs: bool = False,
                                                downsample_field_srcs_fraction: float = 2.0,
                                                drop_missing: bool = True,
                                                random_state: int = 42) -> Dict:
    """
    photo_prior_create_train_test_validation_data creates a train/test/validation data split for prior trainings depending on the data available (for three models)

    Args:
        photo_cat_scaled (pd.DataFrame): training data
        x_ray_flux_bins_num (int, optional): number of bins in X-ray flux. Set this to 1 only! Defaults to 1.
        features_cols (str, optional): one of three models: grzw1w2, grzw1 or grz . Defaults to 'grzw1w2'.
        validation_fraction (float, optional): fraction of data for validation. Defaults to 0.3.
        test_fraction (float, optional): fraction of data for test. Defaults to 0.2.
        downsample_field_srcs (bool, optional): whether to downsample the negative class (label = 0, i.e. field source). Defaults to False.
        downsample_field_srcs_fraction (float, optional): if yes, the downsampling make the negative class this times larger than positive class. Defaults to 2.0.
        drop_missing (bool, optional): whether to drop rows with missing data. Defaults to True.
        random_state (int, optional): random seed. Defaults to 42.

    Returns:
        Dict: train/test/validation data and some info in a dictionary
    """


    if features_cols == 'grzw1w2':
        features_cols = ['rel_dered_mag_g','rel_dered_mag_r','rel_dered_mag_z','rel_dered_mag_w1','rel_dered_mag_w2', 'rel_dered_g_r', 'rel_dered_r_z',  'rel_dered_g_z','rel_dered_z_w1', 'rel_dered_r_w2', 'rel_dered_w1_w2']
    if features_cols == 'grzw1':
        features_cols = ['rel_dered_mag_g','rel_dered_mag_r','rel_dered_mag_z','rel_dered_mag_w1', 'rel_dered_g_r', 'rel_dered_r_z',  'rel_dered_g_z','rel_dered_z_w1']
    elif features_cols == 'grz':
        features_cols = ['rel_dered_mag_g','rel_dered_mag_r','rel_dered_mag_z', 'rel_dered_g_r', 'rel_dered_r_z', 'rel_dered_g_z']
    elif features_cols == 'grzw1w2w3w4':
        features_cols = ['rel_dered_mag_g','rel_dered_mag_r','rel_dered_mag_z','rel_dered_mag_w1','rel_dered_mag_w2', 'rel_dered_mag_w3', 'rel_dered_mag_w4', 'rel_dered_g_r', 'rel_dered_r_z', 'rel_dered_g_z','rel_dered_z_w1', 'rel_dered_r_w2', 'rel_dered_w1_w2', 'rel_dered_z_w3', 'rel_dered_r_w4', 'rel_dered_w3_w4']

    target_col = ['is_counterpart']
    photo_cat = photo_cat_scaled.copy()
    if drop_missing:
        photo_cat.dropna(subset = features_cols, how = 'any', inplace = True)
    else:
        pass
    #assign random number to x_ray_flux_bin for each source which is not a counterpart
    tmp_col = np.random.randint(0, x_ray_flux_bins_num, len(photo_cat))
    photo_cat['x_ray_flux_bin'] = tmp_col
    

    flux_bin_num, flux_bins = pd.qcut(photo_cat[photo_cat.is_counterpart].csc_flux_05_2,  x_ray_flux_bins_num, retbins = True, labels = False)
    photo_cat[photo_cat.is_counterpart]['x_ray_flux_bin'] = flux_bin_num

    print('total x-ray sources: ',len(photo_cat[photo_cat.is_counterpart]))
    print('total non-x-ray sources: ',len(photo_cat[~photo_cat.is_counterpart]))
    print('total sources: ',len(photo_cat))
    print('number of x-ray sources per flux bin:')
    print(photo_cat[photo_cat.is_counterpart].groupby('x_ray_flux_bin').size())
    print('number of non-x-ray sources per flux bin:')
    print(photo_cat[~photo_cat.is_counterpart].groupby('x_ray_flux_bin').size())
    print('x-ray flux bins:')
    print(flux_bins)



    photo_cat_validation, photo_cat_train_test = train_test_split(photo_cat, test_size=1-validation_fraction, stratify = photo_cat[target_col], random_state = random_state)


    output_dict = {}

    for i in range(x_ray_flux_bins_num):
        data_validation = photo_cat_validation[photo_cat_validation['x_ray_flux_bin'] == i]

        X_val = data_validation[features_cols]
        y_val = data_validation[target_col]
        


        data = photo_cat_train_test[photo_cat_train_test['x_ray_flux_bin'] == i]

        n_ctsp = data[data.is_counterpart==1].shape[0]
        n_field = data[data.is_counterpart==0].shape[0]


        if downsample_field_srcs:
            tmp_rat = downsample_field_srcs_fraction*n_ctsp/n_field
            tmp_rat = np.min([tmp_rat, 1])
            data.drop(data[data['is_counterpart'] == 0].sample(frac=1-tmp_rat).index, inplace=True)

            n_ctsp = data[data.is_counterpart==1].shape[0]
            n_field = data[data.is_counterpart==0].shape[0]


        X = data[features_cols]
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, stratify = y, random_state = random_state+1)


        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
        X_val = X_val.to_numpy()
        y_val = y_val.to_numpy()

        print('*'*20)
        print('flux bin: ', flux_bins[i], flux_bins[i+1])

        print('train features: \n ', features_cols)

        print('train size examples - filed: ',  np.sum(~y_train))
        print('train size examples - x-ray:', np.sum(y_train))

        print('test size examples - filed: ',  np.sum(~y_test))
        print('test size examples - x-ray', np.sum(y_test))
        
        print('validation size examples - filed: ',  np.sum(~y_val))
        print('validation size examples - x-ray:', np.sum(y_val))
        
        print('downsampled field sources: ', downsample_field_srcs)
        print('data is scaled')

        pos = np.sum(y_train)
        neg = np.sum(~y_train)
        total = pos + neg
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight_train = {0: weight_for_0, 1: weight_for_1}



        output_dict[i] = {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test, 'X_val':X_val, 'y_val':y_val, 'left_flux_bin': flux_bins[i], 'right_flux_bin': flux_bins[i+1], 'features_cols_scaled': features_cols, 'class_weight_train': class_weight_train}

    return output_dict



def save_keras_classifier(model, hist_df, model_name):
    model.save(model_name+'.keras_nn')
    hist_df.to_csv(model_name+'.csv', index=False, sep = '\t', header = ['#lo', 'hi', 'selected', 'others'])



def find_completeness_purity_intercept(cutoffs: np.ndarray, completeness: np.ndarray, purity: np.ndarray):
    """ Find the completeness and purity at the intersection of the completeness and purity curves."""

    cutoff_intersection_id = np.argmin(np.abs(completeness[completeness>0] - purity[completeness>0]))
    cutoff_intersection = cutoffs[completeness>0][cutoff_intersection_id]
    completeness_intersection = completeness[completeness>0][cutoff_intersection_id]
    purity_intersection = purity[completeness>0][cutoff_intersection_id]

    return cutoff_intersection, completeness_intersection, purity_intersection









def assess_goodnes_of_cross_match(match_df: pd.DataFrame,
                                 match_flag_col: str='nway_match_flag',
                                 candidate_col: str = 'desi_id',
                                 true_ctps_col: str = 'desi_id_true',
                                 calib_col: str = 'nway_prob_has_match',
                                 plot_res: bool = True,
                                 p_any_cut: Optional[float] = None) -> Tuple:
    """
    assess_goodnes_of_cross_match shows the identification matrics for a range of cutoffs: cutoff, overall putiry, completeness, purity, completeness (for hostless), purity (for hostless), all at a given cutoff

    Args:
        match_df (pd.DataFrame): catalog with cross-match results and true id of counterpart (or if is is hostless)
        match_flag_col (str, optional): flag to filter selected counterparts based on cross-match (e.g. the one with highest p_i). Defaults to 'nway_match_flag'.
        candidate_col (str, optional): column name with id of predicted counterpart. Defaults to 'desi_id'.
        true_ctps_col (str, optional): column name with id of true counterpart. Defaults to 'desi_id_true'.
        calib_col (str, optional): parameter to calibrate cutoffs. Defaults to 'nway_prob_has_match'.
        plot_res (bool, optional): whether to plot resulting curves. Defaults to True.
        p_any_cut (Optional[float], optional): whether to apply some cut beforehand (print statistics there). Defaults to None.

    Returns:
        Tuple: see above 
    """


    match_df_orig  = match_df.copy()
    match_df = match_df.copy()
    match_df = match_df[match_df[match_flag_col]==1]
    match_df = match_df.query(f"~{true_ctps_col}.isna()")


    n_ctps = len(match_df)
    n_ctps_not_hostless = len(match_df[match_df[true_ctps_col]!='hostless'])
    n_ctps_hostless = n_ctps - n_ctps_not_hostless

    def make_cut(match_df_in, cutoff):
        match_df_out = match_df_in.copy()
        cut_mask = match_df_out.eval(f"{calib_col}>{cutoff}")
        match_df_out.loc[~cut_mask, candidate_col] = 'hostless'

        return match_df_out


    def calc_stats(match_df, match_df_orig,  verbose = False):
        match_df = match_df.copy()
        total_hostless = len(match_df_orig[match_df_orig[true_ctps_col]=='hostless'])
        total_not_hostless = len(match_df_orig[match_df_orig[true_ctps_col]!='hostless'])

        n_correct_not_hostless_and_correct_ctp = len(match_df.query(f"{true_ctps_col}=={candidate_col} & {true_ctps_col}!='hostless'"))

        n_correct_not_hostless_and_incorrect_ctp = len(match_df.query(f"{true_ctps_col}!={candidate_col} & {true_ctps_col}!='hostless' & {candidate_col}!='hostless'"))

        n_incorrect_not_hostless = len(match_df.query(f"{true_ctps_col}=='hostless' & {candidate_col}!='hostless'"))

        n_incorrect_hostless = len(match_df.query(f"{true_ctps_col}!='hostless' & {candidate_col}=='hostless'"))

        n_correct_hostless = len(match_df.query(f"{true_ctps_col}=='hostless' & {candidate_col}=='hostless'"))



        overall_purity = (n_correct_hostless+n_correct_not_hostless_and_correct_ctp) / (total_hostless + total_not_hostless)

        not_hostless_purity = (n_correct_not_hostless_and_correct_ctp) / ( n_correct_not_hostless_and_correct_ctp+ n_correct_not_hostless_and_incorrect_ctp + n_incorrect_not_hostless)

        not_hostless_completeness = (n_correct_not_hostless_and_correct_ctp) / (total_not_hostless)

        hostless_purity = (n_correct_hostless) / (n_correct_hostless + n_incorrect_hostless)

        hostless_completeness = (n_correct_hostless) / (total_hostless)


        if verbose:
            print(f"""
                Total validation set: {total_hostless + total_not_hostless}
                \t Total hostless: {total_hostless}
                \t Total not hostless: {total_not_hostless}
                Assigned not hostless:
                \t Correct not hostless with correct ctp: {n_correct_not_hostless_and_correct_ctp}
                \t Correct not hostless with incorrect ctp: {n_correct_not_hostless_and_incorrect_ctp}
                \t Incorrect not hostless: {n_incorrect_not_hostless}
                Assigned hostless:
                \t Incorrect hostless: {n_incorrect_hostless}
                \t Correct hostless: {n_correct_hostless}

                Overall purity: {overall_purity:.2f}
                Not hostless purity: {not_hostless_purity:.2f}
                Not hostless completeness: {not_hostless_completeness:.2f}
                Hostless purity: {hostless_purity:.2f}
                Hostless completeness: {hostless_completeness:.2f}
            """) 


        return overall_purity, not_hostless_purity, not_hostless_completeness, hostless_purity, hostless_completeness
        
    overall_purity_arr = []
    purity_not_hostless_arr = []
    completeness_not_hostless_arr = []
    purity_hostless_arr = []
    completeness_hostless_arr = []

    cutoffs = np.linspace(0.02,0.98,100)
    for cutoff in cutoffs:
        match_df_cut = make_cut(match_df, cutoff)
        stat_res = calc_stats(match_df_cut, match_df, verbose=False)
        overall_purity, purity_not_hostless, completeness_not_hostless, purity_hostless, completeness_hostless = stat_res

        purity_not_hostless_arr.append(purity_not_hostless)
        completeness_not_hostless_arr.append(completeness_not_hostless)
        purity_hostless_arr.append(purity_hostless)
        completeness_hostless_arr.append(completeness_hostless)
        overall_purity_arr.append(overall_purity)

    purity_not_hostless_arr = np.array(purity_not_hostless_arr)
    completeness_not_hostless_arr = np.array(completeness_not_hostless_arr)
    purity_hostless_arr = np.array(purity_hostless_arr)
    completeness_hostless_arr = np.array(completeness_hostless_arr)
    overall_purity_arr = np.array(overall_purity_arr)


    cutoff_intersection, completeness_intersection, purity_intersection = find_completeness_purity_intercept(cutoffs, completeness_not_hostless_arr, purity_not_hostless_arr)



    frac_src_p_any_over = (match_df_orig[calib_col] > cutoff_intersection ).astype(int).mean()
    frac_src_p_any_over = np.round(frac_src_p_any_over*100, 2)

    if plot_res:
        plt.figure(figsize=(9,6))
        plt.plot(cutoffs, overall_purity_arr, 'C0', label='Overall purity')
        plt.plot(cutoffs, purity_not_hostless_arr, 'C1-',  label='Purity not hostless')
        plt.plot(cutoffs, completeness_not_hostless_arr, 'C1--', label='Completeness not hostless')
        plt.plot(cutoffs, purity_hostless_arr, 'C2-', label='Purity hostless')
        plt.plot(cutoffs, completeness_hostless_arr, 'C2--', label='Completeness hostless')

        plt.legend()
        plt.xlabel(calib_col+' cutoff')
        plt.ylabel('purity/completeness')

        plt.axvline(cutoff_intersection, color='k', ls='--', label=f'purity=completeness={completeness_intersection:.2g}%; \n {frac_src_p_any_over:.2g}% of sources have prob_has_match > {cutoff_intersection:.2g}')


        print(f" Completeness = {100*completeness_intersection:.2g}% \n Purity = {100*purity_intersection:.2g}% \n {calib_col} optimal cutoff =  {cutoff_intersection:.2g} \n Fraction of sources with prob_has_match > {cutoff_intersection:.2g} = {frac_src_p_any_over:.2g}%")
        #plt.ylim(0, 1.1)
        plt.ylim(0.6, 1.1)
        plt.show()
    else:
        pass

    if p_any_cut is None:
        print(f'p_any cut: {cutoff_intersection:.2g}')
        calc_stats(make_cut(match_df, cutoff_intersection), match_df, verbose = True)
    else:
        print(f'p_any cut: {p_any_cut:.2g}')
        calc_stats(make_cut(match_df, p_any_cut), match_df, verbose = True)

    return cutoffs, overall_purity_arr, completeness_not_hostless_arr, purity_not_hostless_arr, completeness_hostless_arr, purity_hostless_arr, match_df



def flux2mag(flux):
    #converts flux to magnitude
    return 22.5 - 2.5 * np.log10(flux)

def flux_nmagg2vega_mag(flux:pd.Series,
                        mode:str) -> pd.Series:
    """
    Converts DESI w1 flux (in nanomaggies) to
    vega magnitudes.
    
    https://www.legacysurvey.org/dr9/description/
    """
    if mode=='w1':
        delta_m = 2.699
    elif mode=='w2':
        delta_m = 3.339
    elif mode=='w3':
        delta_m = 5.174
    elif mode=='w4':
        delta_m = 6.620
    else:
        raise ValueError('Mode must be one of w1, w2, w3, w4')
    
    vega_flux = flux * 10 ** (delta_m / 2.5)
    vega_mag = flux2mag(vega_flux)
    vega_mag = vega_mag.replace([np.inf, -np.inf], np.nan)
    
    return vega_mag


def flux_frequency_correction(magnitudes: pd.Series,
                              w_eff: float,
                              ab_zeropoint: float) -> pd.Series:
    """
    Converts magnitudes obtainded from nanomaggies (erg/cm²/Hz)
    to flux in erg/cm²/s.
    http://svo2.cab.inta-csic.es/theory/fps/index.php?id=CTIO/DECam.z&&mode=browse&gname=CTIO&gname2=DECam#filter
    Args:
        magnitudes (pd.Series): Magnitudes in AB system.
        w_eff (float): width of the effective wavelength.
        ab_zeropoint (float): Zero Point in AB System.
    Returns:
        pd.Series: corrected flux
    """

    flux = w_eff * ab_zeropoint * 10 ** (-0.4 * magnitudes)
    flux.name = 'flux_corrected'

    return flux


def desi_reliable_magnitudes(df: pd.DataFrame,
                        s_n_threshold: int = 4,
                        colors: bool=True,
                        prefix: str = '',
                        ) -> pd.DataFrame:
    """
    sources: https://github.com/mbelveder/luminosity_LH/blob/9f4837cb509c780e1c3db79b05c9cc0cd4932c2c/lh_functions.py#L372
    Calculate reliable magnitudes only for objects with reliable flux measurments.
    
    https://www.legacysurvey.org/dr9/description/:
    "The fluxes can be negative for faint objects, and indeed we expect
    many such cases for the faintest objects."
    Args:
        df (pd.DataFrame): DESI catalogue.
        s_n_threshold (int): S/N threshold.
        colors (bool): If True, calculate colors.
        prefix (str): Prefix for the new columns. It also assumes that you want to add a prefix to a dataframe with (DESI) columns that already have a prefix.
    Returns:
        pd.DataFrame: Catalogue with reliable magnitudes.
    """
    np.seterr(divide = 'ignore')  #ignore /0 errors in log10
    df = df.copy()
    original_columns = df.columns

    for band in ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:

        flux_colname = prefix+f'flux_{band}'
        flux_ivar_colname = prefix+f'flux_ivar_{band}'
        dered_mag_colname = prefix+f'dered_mag_{band}'


        # All magnitudes (unreliable included)
        df[f'all_mag_{band}'] = flux2mag(df[flux_colname])
        
        # Select only reliable fluxes (allows avoiding correction on noise)
        flux_sn = df[flux_colname].abs() * np.sqrt(df[flux_ivar_colname])
        reliable_flux = pd.Series(np.where(flux_sn > s_n_threshold, df[flux_colname], np.nan))
        # df[f'rel_flux_{band}'] = reliable_flux

        # Calculate reliable magnitudes
        df[f'rel_mag_{band}'] = flux2mag(reliable_flux)
        df[f'rel_mag_{band}'] = df[f'rel_mag_{band}'].replace([np.inf, -np.inf], np.nan)

        # Reliable dereddended magnitudes
        df[f'rel_dered_mag_{band}'] = np.where(
            flux_sn > s_n_threshold, df[dered_mag_colname], np.nan
            )

        # Reliable Vega magnitudes for WISE fluxes
        if 'w' in band:
            df[f'vega_mag_{band}'] = flux_nmagg2vega_mag(reliable_flux, mode=band)



        W_EFF_G = 1204.22
        AB_ZEROPOINT_G = 4.78525e-9

        df['rel_flux_corr_g'] = flux_frequency_correction(
                df['rel_dered_mag_g'],
                w_eff=W_EFF_G,
                ab_zeropoint=AB_ZEROPOINT_G
                )


    W_EFF_R = 1311.48
    AB_ZEROPOINT_R = 2.66574e-9

    df['rel_flux_corr_r'] = flux_frequency_correction(
            df['rel_dered_mag_r'],
            w_eff=W_EFF_R,
            ab_zeropoint=AB_ZEROPOINT_R
            )

    W_EFF_Z = 1291.48
    AB_ZEROPOINT_Z = 1.286e-9

    df['rel_flux_corr_z'] = flux_frequency_correction(
            df['rel_dered_mag_z'],
            w_eff=W_EFF_Z,
            ab_zeropoint=AB_ZEROPOINT_Z
            )



    if colors:
        #g-z, g-z, r-z all dered
        df['rel_dered_g_r'] = df['rel_dered_mag_g'] - df['rel_dered_mag_r']
        df['rel_dered_g_z'] = df['rel_dered_mag_g'] - df['rel_dered_mag_z']
        df['rel_dered_r_z'] = df['rel_dered_mag_r'] - df['rel_dered_mag_z']


        #z-w1, r-w2, w1-w2 all dered
        df['rel_dered_z_w1'] = df['rel_dered_mag_z'] - df['rel_dered_mag_w1']
        df['rel_dered_r_w2'] = df['rel_dered_mag_r'] - df['rel_dered_mag_w2']
        df['rel_dered_w1_w2'] = df['rel_dered_mag_w1'] - df['rel_dered_mag_w2']
        

        #z-w3, r-w4, w3-w4 all dered
        df['rel_dered_z_w3'] = df['rel_dered_mag_z'] - df['rel_dered_mag_w3']
        df['rel_dered_r_w4'] = df['rel_dered_mag_r'] - df['rel_dered_mag_w4']
        df['rel_dered_w3_w4'] = df['rel_dered_mag_w3'] - df['rel_dered_mag_w4']




    xray = 'flux_05-20' in original_columns
    if xray:
        # X-ray to optical flux
        df['lg(Fx/Fo_g)'] = np.log10(df['flux_05-20'] / df[prefix+'flux_g'])
        df['lg(Fx/Fo_r)'] = np.log10(df['flux_05-20'] / df[prefix+'flux_r'])
        df['lg(Fx/Fo_z)'] = np.log10(df['flux_05-20'] / df[prefix+'flux_z'])


        dered_flux_z = 10 ** (9 - df['rel_dered_mag_z'] / 2.5)
        df['rel_dered_lg(Fx/Fo_z)'] = np.log10(df['flux_05-20'] / dered_flux_z)
        df['rel_dered_lg(Fx/Fo_z_corr)'] = np.log10(df['flux_05-20'] / df['rel_flux_corr_z'])

        dered_flux_g = 10 ** (9 - df['rel_dered_mag_g'] / 2.5)
        df['rel_dered_lg(Fx/Fo_g)'] = np.log10(df['flux_05-20'] / dered_flux_g)
        df['rel_dered_lg(Fx/Fo_g_corr)'] = np.log10(df['flux_05-20'] / df['rel_dered_mag_g'])

        dered_flux_r = 10 ** (9 - df['rel_dered_mag_r'] / 2.5)
        df['rel_dered_lg(Fx/Fo_r)'] = np.log10(df['flux_05-20'] / dered_flux_r)
        df['rel_dered_lg(Fx/Fo_r_corr)'] = np.log10(df['flux_05-20'] / df['rel_flux_corr_r'])


    new_cols = [col for col in df.columns if col not in original_columns]
    new_cols_renamed = [prefix + col for col in new_cols]

    df = df.rename(columns=dict(zip(new_cols, new_cols_renamed)))

    return df



def rayleigh_plot(input_cross_match_df: pd.DataFrame,  
                    sep_col: str = 'sep', pos_err_col: str = 'pos_err',
                    plotlabel: str = 'eROSITA',
                    ylim: Tuple=(1e-3, 1)): 
    """
    rayleigh_plot make a plot of distirubtion of separation divided by positional error, which is expected to be Rayleigh distributed (0, 1).
    all cuts and queries on the primary catalog (e.g. Detection Likelihood) should be done before calling this function.
    Args:
        input_cross_match_df (pd.DataFrame): Input dataframe with cross-matched catalog 1 (e.g measured positions with eROSITA) with catalog 2 (true positions, e.g. DESI counterparts).
        sep_col (str, optional): name of the columns with separation between the two. Defaults to 'sep'.
        pos_err_col (str, optional): column with positional error (1 sigma). Defaults to 'pos_err'.
        plotlabel (str, optional): Label of the plot. Defaults to 'eROSITA'.
        ylim (Tuple, optional): Limits on y axis. Defaults to (1e-3, 1).
    """


    input_cross_match_df = input_cross_match_df.copy()



    rat = input_cross_match_df[sep_col]/input_cross_match_df[pos_err_col]
    input_cross_match_df['rat'] = rat
    rayleigh_fit = stats.rayleigh.fit(rat)


    fig, axs =  plt.subplots(nrows=2, ncols = 1, sharex = True, gridspec_kw = {'hspace':0, 'height_ratios': None}, figsize = (12,12))
    ax, ax2 = axs
    
    sns.ecdfplot(data = input_cross_match_df, x = 'rat', ax = ax, complementary = True, lw = 3)
    sns.histplot(input_cross_match_df, x = 'rat', ax = ax2, stat = 'density', lw = 3, bins = 50)

    for prob in [39.3, 68, 95, 98]:
        ax.axhline(1 - prob/100, color = 'k', ls = '--', alpha = 0.5)
        ax.text(0.5, 1 - prob/100, f'{prob}%', ha = 'center', va = 'center', color = 'k', alpha = 0.5)


    #plot the fit
    x = np.linspace(0, rat.max()*1.05, 100)
    ax.plot(x, 1-stats.rayleigh.cdf(x, *rayleigh_fit), 'r-', lw=3, alpha=0.6, label='Rayleigh fit: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % rayleigh_fit, zorder = -1)
    ax.plot(x, 1-stats.rayleigh.cdf(x, 0,1), 'g-', lw=3, alpha=0.6, label='Rayleigh fixed: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % (0,1), zorder = -1)
    ax.set(ylim=ylim)
    ax2.set_xlabel('Separation/pos_err')
    ax.set_yscale('log')

    ax2.plot(x, stats.rayleigh.pdf(x, *rayleigh_fit), 'r-', lw=3, alpha=0.6, label='Rayleigh fit: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % rayleigh_fit, zorder = -1)
    ax2.plot(x, stats.rayleigh.pdf(x, 0,1), 'g-', lw=3, alpha=0.6, label='Rayleigh fixed: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % (0,1), zorder = -1)

    plt.legend()
    plt.suptitle(plotlabel+', '+str(len(input_cross_match_df))+' sources')


def add_separation_columns(df: pd.DataFrame, 
                            colname_ra1: str, colname_dec1: str,
                            colname_ra2: str, colname_dec2: str,
                            colname: str = 'sep') -> pd.DataFrame:
    """
    add_separation_columns adds a column with separation between two sets of coordinates in degrees in one dataframe (e.g. dataframe with X-ray coordinates and the coordinates of counterparts)

    Args:
        df (pd.DataFrame): Dataframe with both coordinates
        colname_ra1 (str): columns name for RA of the first set of coordinates, in degrees
        colname_dec1 (str): --||-- DEC --||--
        colname_ra2 (str): --||-- RA --||-- of the second set of coordinates, in degrees
        colname_dec2 (str): --||-- DEC --||--
        colname (str, optional): name of separation column to add to the dataframe. Defaults to 'sep'.

    Returns:
        pd.DataFrame: modified version of dataframe
    """

    df = df.copy()


    coords1 = SkyCoord(ra = df[colname_ra1].values*u.degree, dec = df[colname_dec1].values*u.degree)
    coords2 = SkyCoord(ra = df[colname_ra2].values*u.degree, dec = df[colname_dec2].values*u.degree)

    seps = coords1.separation(coords2)
    seps = seps.to(u.arcsec).value

    df[colname] = seps
    return df
     





def cross_match_data_frames(df1: pd.DataFrame, df2: pd.DataFrame, 
                            colname_ra1: str, colname_dec1: str,
                            colname_ra2: str, colname_dec2: str,
                            match_radius: float = 3.0,
                            df_prefix: str = '',
                            closest: bool = False,
                            ) -> pd.DataFrame:
    """
    cross_match_data_frames cross-matches two dataframes.
    Cross-match two dataframes with astropy
    https://docs.astropy.org/en/stable/api/astropy.coordinates.match_coordinates_sky.html#astropy.coordinates.match_coordinates_sky
    https://docs.astropy.org/en/stable/api/astropy.coordinates.search_around_sky.html#astropy.coordinates.search_around_sky
    Args:
        df1 (pd.DataFrame): first catalog
        df2 (pd.DataFrame): second catalog
        colname_ra1 (str): columns name for ra in df1, in degrees
        colname_dec1 (str): columns name for dec in df1, in degrees
        colname_ra2 (str): columns name for ra in df2, in degrees
        colname_dec2 (str): columns name for dec in df2, in degrees
        match_radius (float, optional): match radius in arcsec. Defaults to 3.0.
        df_prefix (str, optional): prefix to prepend to the columns of the second data frame. Defaults to ''. If not '', '_' is prepended.
        closest (bool, optional): whether to return the closest match. Defaults to False.

    Returns:
        pd.DataFrame: match of df1 and df2

        the columns are from the original df1 and df2 (with the prefix for df2). 
        added columns: 
        sep - separation in arcsec
        
        n_near - number of matches from df2  for a particular source from df1. For example n_near=10 for a source in df1 means that there are 10 sources  in df2 within the match_radius.

        n_matches - for a given source from df2, this is a number of sources from df1 that are within match_radius. For example n_matches = 2 for a source in df2 means that there are 2 sources in df1 which have this source from df2 within match_radius. If n_matches = 1 then this source from df2 is unique.


    example:
    cross_match_data_frames(desi, gaia, 
                                colname_ra1='RA',
                                colname_dec1='DEC',
                                colname_ra2='ra',
                                colname_dec2='dec',
                                match_radius = 10,
                                df_prefix = 'GAIA',
                                closest=False)
    """
    if df_prefix != '':
        df_prefix = df_prefix + '_'
    else:
        df_prefix = ''

    df1 = df1.copy()
    df2 = df2.copy()

    orig_size_1 = df1.shape[0]
    orig_size_2 = df2.shape[0]

    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df1.rename(columns={'index': 'index_primary'}, inplace=True)
    df2.rename(columns={'index': 'index_secondary'}, inplace=True)


    coords1 = SkyCoord(ra = df1[colname_ra1].values*u.degree, dec = df1[colname_dec1].values*u.degree)
    coords2 = SkyCoord(ra = df2[colname_ra2].values*u.degree, dec = df2[colname_dec2].values*u.degree)

    idx1, idx2, ang_sep, _ = coordinates.search_around_sky(coords1, coords2, match_radius*u.arcsec)
    ang_sep = ang_sep.to(u.arcsec)
    ang_sep = pd.DataFrame({df_prefix+'sep': ang_sep})

    df1 = df1.loc[idx1]
    df2 = df2.loc[idx2]

    df1.reset_index(inplace = True, drop = True)
    df2.reset_index(inplace = True, drop = True)


    df2.columns  = [df_prefix+x for x in df2.columns]
    df2.rename(columns={df_prefix+'index_secondary':'index_secondary'}, inplace=True)

    df_matched = pd.concat([df1, df2, ang_sep], axis=1) 



    df_matched.sort_values(by=['index_primary', df_prefix+'sep'], inplace=True, ascending=True)

    

    df_matched[df_prefix+'n_near'] = df_matched.groupby('index_primary')[df_prefix+'sep'].transform('count')

    second_index_value_counts = df_matched['index_secondary'].value_counts()
    df_matched[df_prefix+ 'n_matches'] = df_matched['index_secondary'].apply(lambda x: second_index_value_counts[x])


    print('cross-match radius', match_radius, 'arcsec')
    print('total matches:', len(df_matched), 'out of', orig_size_1, 'x' ,orig_size_2)

    print('\t total unique pairs:', len(df_matched.query(df_prefix+'n_matches == 1')))
    
    print('\t total non-unique pairs (duplicates in df2):', len(df_matched.query(df_prefix+'n_matches > 1')))

    if closest:
        df_matched = df_matched.drop_duplicates(subset=['index_primary'], keep='first')
        print('total closest matches:', len(df_matched))

    df_matched.drop(columns=['index_primary'], inplace=True)
    df_matched.drop(columns=['index_secondary'], inplace=True)

    return df_matched                  




def search_around_r_data_frames(df1: pd.DataFrame, target_ra: float, target_dec: float,
                            colname_ra1: str, colname_dec1: str,
                            match_radius: float = 3.0,
                            closest: bool = False,
                            ) -> pd.DataFrame:
    """the same as cross_match_data_frames but for a single target (target_ra, target_dec)"""

    df2 = pd.DataFrame({'RA': [target_ra], 'DEC': [target_dec]})
    df_matched = cross_match_data_frames(df1, df2,
                                colname_ra1=colname_ra1,
                                colname_dec1=colname_dec1,
                                colname_ra2='RA',
                                colname_dec2='DEC',
                                match_radius=match_radius,
                                df_prefix='',
                                closest=closest,
                                )

    return df_matched                  


def prepare_nway_results(nway_res_orig: pd.DataFrame,
                        ero_for_nway_fits: str = "ERO_lhpv_03_23_sd01_a15_g14.fits",
                        desi_for_nway_fits: str = "desi_lh.fits",
                        ero_full_cat: str = 'ERO_lhpv_03_23_sd01_a15_g14.pkl',
                        desi_full_cat: str = 'desi_lh.gz_pkl',
                        ero_desi_ctps_file: str = 'validation_ctps_ero_desi_lh.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    prepare_nway_results process results from NWAY output and prepares the final result of cross-match.
    If the true match is known and NWAY is incorrect, the true match is added to the result instead of the NWAY match!

    Args:
        nway_res_orig (pd.DataFrame): dataFrame with NWAY results
        ero_for_nway_fits (str, optional): fits file used as NWAY input for X-ray catalog. Defaults to "ERO_lhpv_03_23_sd01_a15_g14.fits".
        desi_for_nway_fits (str, optional): fits file used as NWAY input for the secondary catalog (DESI LIS). Defaults to "desi_lh.fits".
        ero_full_cat (str, optional): Full x-ray catalog, to append X-ray properties to the results. Defaults to 'ERO_lhpv_03_23_sd01_a15_g14.pkl'.
        desi_full_cat (str, optional): Full optical catalog to append optical properties to the result. Defaults to 'desi_lh.gz_pkl'.
        ero_desi_ctps_file (str, optional): File with a list of true counterparts to x-ray sources (e.g. DESI counterparts and hostless sources). Defaults to 'validation_ctps_ero_desi_lh.csv'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: data frame of results for all matches and data frame of results for the best match
    """


    nway_res_orig = nway_res_orig.copy()
    nway_res_orig = nway_res_orig[nway_res_orig.DESI!=-1] #nway puts a row with DESI = -1 for every source of the primary match

    #load files used in NWAY
    ero_pandas = fits_to_pandas(ero_for_nway_fits)
    desi_pandas = fits_to_pandas(desi_for_nway_fits)

    #Load original files with full catalogs
    erosita_orig_df = pd.read_pickle(data_path+ero_full_cat)
    desi_orig_df =  pd.read_pickle(data_path+desi_full_cat, compression = 'gzip')
    desi_orig_df.columns = ['desi_'+x for x in desi_orig_df.columns]
    desi_orig_df.rename(columns={'desi_desi_id':'desi_id'}, inplace=True)


    ero_true_ctps = pd.read_csv(data_path+ero_desi_ctps_file)
    erosita_orig_df = erosita_orig_df.merge(ero_true_ctps, on='srcname_fin', how='left')



    #add a prefix nway_ to the column with match parameters
    nway_res = nway_res_orig.rename(columns={c: 'nway_'+c for c in nway_res_orig.
    columns if c not in ['EROSITA', 'DESI']})


    #assign srcname_fin as a ID from ero_pandas according to index in nway_res_orig
    nway_res['srcname_fin']  = ero_pandas['ID'].values[nway_res['EROSITA']]
    #the same for desi_pandas
    nway_res['desi_id'] = desi_pandas['desi_id'].values[nway_res['DESI']]

    #make a table with photometry-only data from DESI dataframe (i.e. all columns are scaled and replaced with -99 if used/not-available/non-significant)
    desi_pandas_photometry = desi_pandas.drop(['DESI','ra', 'dec'], axis=1, inplace=False)
    #rename the columns to avoid confusion with further full desi data
    desi_pandas_photometry = desi_pandas_photometry.rename(columns={c: 'nway_photometry_'+c for c in desi_pandas_photometry.columns if c not in ['desi_id']})
    #join the photometry-only data to nway_res
    nway_res = nway_res.merge(desi_pandas_photometry, on='desi_id', how='left')

    #drop EROSITA/DESI columns as they are not needed anymore
    nway_res = nway_res.drop(['EROSITA', 'DESI'], axis=1, inplace=False)

    #merge with eROSITA data
    nway_res = erosita_orig_df.merge(nway_res, on = 'srcname_fin')

    #merge with DESI data
    nway_res = nway_res.merge(desi_orig_df, on = 'desi_id')


    #add additional DESI columns including X-ray to optical ratio
    nway_res = desi_reliable_magnitudes(nway_res, s_n_threshold=3, prefix='desi_')


    #assigning match flags

    def get_flag_num(x, flag):
        try:
            return x.value_counts()[flag]
        except:
            return 0


    nway_res['nway_n_match_flag_0'] = nway_res.groupby('srcname_fin')['nway_match_flag'].transform(lambda x: get_flag_num(x, 0))

    nway_res['nway_n_match_flag_2'] = nway_res.groupby('srcname_fin')['nway_match_flag'].transform(lambda x: get_flag_num(x, 2))

    tmp = nway_res.groupby('srcname_fin')['nway_Separation_EROSITA_DESI'].transform(lambda x: min(x))
    nway_res['nway_is_closest'] = tmp == nway_res['nway_Separation_EROSITA_DESI']

    nway_res['nway_is_within_pos_r98'] = nway_res['nway_Separation_EROSITA_DESI'] < nway_res['pos_r98'] #

    #nway_res['nway_closest_is_psf'] = nway_res.groupby('srcname_fin')['desi_type'].transform(lambda x: x.iloc[0] == 'PSF') #dont forget to sort by separation first


    nway_res.sort_values(by=['srcname_fin', 'nway_prob_this_match' ], inplace=True, ascending=[True, False])

    if 'nway_bias_DESI_nnmag_grz' not in nway_res.columns:
        cols_to_drop = ['nway_dist_bayesfactor_uncorrected', 'nway_dist_bayesfactor', 'nway_dist_post', 'nway_Separation_max', 'nway_ncat', 'nway_p_single']
        nway_res.drop(columns= cols_to_drop, axis=1, inplace=True)
    else:
        cols_to_drop =  ['nway_dist_bayesfactor_uncorrected', 'nway_dist_bayesfactor', 'nway_dist_post', 'nway_Separation_max', 'nway_ncat', 'nway_p_single', 'nway_bias_DESI_nnmag_grz', 'nway_bias_DESI_nnmag_grzw1', 'nway_bias_DESI_nnmag_grzw1w2', 'nway_bias_DESI_rel_dered_mag_g', 'nway_bias_DESI_rel_dered_mag_r', 'nway_bias_DESI_rel_dered_mag_z', 'nway_bias_DESI_rel_dered_g_r', 'nway_bias_DESI_rel_dered_r_z', 'nway_bias_DESI_rel_dered_g_z']
        nway_res.drop(columns = cols_to_drop, axis=1, inplace=True)



    #quick test that everything is ok on a sample of a few sources
    check_cols = ['prob_has_match', 'prob_this_match', 'Separation_EROSITA_DESI', 'match_flag'] 

    for _ in range(10):
        random_srcname = nway_res['srcname_fin'].sample(1).values[0]
        random_desi_id_for_srcname = nway_res.loc[nway_res['srcname_fin'] == random_srcname, 'desi_id'].values[0]

        id_of_srcname = ero_pandas.query("ID == @random_srcname").index.values[0]
        id_of_desi_id = desi_pandas.query("desi_id == @random_desi_id_for_srcname").index.values[0]

        my_res = nway_res.query("srcname_fin == @random_srcname & desi_id == @random_desi_id_for_srcname")

        expected_res = nway_res_orig.query("EROSITA==@id_of_srcname  & DESI == @id_of_desi_id")

        for col in check_cols:

            assert my_res['nway_'+col].values[0] == expected_res[col].values[0], f"nway_{col} does not match for srcname {random_srcname} and desi_id {random_desi_id_for_srcname}"

    print('conjugation test passed')



    #now we prepare a data frame with only the best match for each eROSITA source
    #and take into account the true counterpart of each eROSITA source if it is available. If nway assigns incorrect match, I will use the true counterpart instead of the counterpart assigned by nway

    nway_res_best = nway_res.copy()

    #step 1: find indeces of incorrect matches. i.e. match_flag==1 but desi_id!=desi_id_true_ctp
    idx_incorrect = nway_res_best.eval('nway_match_flag==1 & desi_id!=desi_id_true & desi_id_true!="hostless" & ~desi_id_true.isna()').values
    nway_res_best[idx_incorrect][['srcname_fin', 'desi_id', 'desi_id_true', 'nway_match_flag']]
    total_valid = nway_res_best.query('desi_id_true!="hostless" & ~desi_id_true.isna()').desi_id_true.nunique()
    print('number of incorrect matches: ', idx_incorrect.sum(), ' out of ', total_valid, ' validation sources')


    #step 2: assign nway_match_flag=2 to all incorrect matches
    nway_res_best.loc[idx_incorrect, 'nway_match_flag'] = 2
    print('assigning nway_match_flag=2 to all incorrect matches')

    #step 3: assign nway_match_flag=1 to the corresponding correct pairs desi_id -- desi_id_true
    idx_incorrect = nway_res_best.eval('desi_id==desi_id_true & desi_id_true!="hostless" & ~desi_id_true.isna()').values
    nway_res_best.loc[idx_incorrect, 'nway_match_flag'] = 1
    print('assigning nway_match_flag=1 to the corresponding correct pairs desi_id -- desi_id_true')


    #step 4: sanity check
    n_unique_true_ctps = nway_res_best.query('desi_id_true!="hostless" & ~desi_id_true.isna()').desi_id_true.nunique()
    n_correct_matches = len(nway_res_best.query('nway_match_flag==1 &  desi_id==desi_id_true & desi_id_true!="hostless" & ~desi_id_true.isna()'))
    n_incorrect_matches = len(nway_res_best.query('nway_match_flag==1 &  desi_id!=desi_id_true & desi_id_true!="hostless" & ~desi_id_true.isna()'))

    assert n_unique_true_ctps == n_correct_matches 
    assert n_incorrect_matches == 0

    nway_res_best = nway_res_best.query('nway_match_flag == 1 ')

    nway_res.sort_values(by=['srcname_fin', 'nway_prob_this_match'], ascending=[True, False], inplace=True)
    nway_res_best = nway_res_best.sort_values(by=['srcname_fin', 'nway_prob_this_match'], ascending=[True, False])

    nway_res.reset_index(drop=True, inplace=True)
    nway_res_best.reset_index(drop=True, inplace=True)



    return nway_res, nway_res_best


