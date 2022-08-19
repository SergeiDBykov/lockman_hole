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

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
import tensorflow as tf
pd.options.mode.chained_assignment = None

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
    Creates healpix maps of quantity observed at ra, dec (in degrees) by taking
    the mean or sum of quantity in each pixel.
    source: https://github.com/xuod/castor/blob/master/castor/cosmo.py
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


def decode_str_columns(df):
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


def fits_to_pandas(filename: str,):
    data = Table.read(data_path+filename, format='fits')
    with fits.open(data_path+'/'+filename , 'readonly') as file:
        dataname = file[1].name
    #the next is to handle multi-dimensional columns
    names = [name for name in data.colnames if len(data[name].shape) <= 1]
    dataframe = data[names].to_pandas()
    dataframe.reset_index(inplace=True)
    dataframe.rename(columns={'index': dataname}, inplace=True)

    return dataframe


def my_scaler_forward(df):
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

def my_scaler_backward(df_scaled):
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


def assess_classifier(clf, X_test, y_test, label = 'Validation set', histbins = 30):  
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


    # try:
    #     ax = sns.displot(x = predict_proba, hue = y_test, bins = 50, stat='density')
    # except:
    #     y_test = y_test[:,0]
    #     ax = sns.displot(x = predict_proba, hue = y_test, bins = 50, stat='density')
    # ax.set(title = label, ylabel  = 'probability density', xlabel = 'classifier predicted probability')



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






    # fig,  ax =  plt.subplots( figsize = (8,8))
    # ax.plot(precision, recall, label=label, linewidth=2)
    # ax.set_xlabel('Precision/completeness')
    # ax.set_ylabel('Recall/putiry')
    # plt.grid(True)
    # ax.set_aspect('equal')
    # id_optim = np.argmin(np.abs(precision-recall))
    # threshold_optim = thresholds[id_optim]
    # precision_optim = precision[id_optim]
    # recall_optim = recall[id_optim]
    # print('Optimal threshold: {:.2f}'.format(threshold_optim))
    # print('Optimal precision: {:.2f}'.format(precision_optim))
    # ax.plot(precision_optim, recall_optim, 'o', color='C5', label = f"({precision_optim:.2f},{recall_optim:.2f})")
    # ax.plot([0,1], [0,1], '--', color='C4')

    # cm = sklearn.metrics.confusion_matrix(y_test, predict_proba > threshold_optim)

    # cm_str = f"TN: {cm[0,0]}  FP: {cm[0,1]}\nFN: {cm[1,0]}  TP: {cm[1,1]}"
    # #add a string version of the confusion matrix, add true positive , false positive  etc labels

    # ax.text(0.3, 0.15, cm_str, ha='center', va='center', transform=ax.transAxes)
    # ax.legend()


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

    #save histograms in one file:
    #with columns lo, hi, selected, others
    hist_df = pd.DataFrame({'lo':bin_field[:-1], 'hi':bin_field[1:], 'selected':hist_ctsp, 'others':hist_field})



    return threshold_optim, precision_optim, recall_optim, hist_df






def plot_metrics(history, metrics = ['loss', 'purity', 'completeness']):
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



def build_keras_model(input_features_shape,
                        activation='relu', 
                        layers_num = (8,8,8),
                        dropout_rate = 0.0,
                        initial_bias = None,
                        lr = 1e-3,
                        load_weights = True,):
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



def photo_prior_create_train_test_validation_data(photo_cat_scaled, x_ray_flux_bins_num = 1, features_cols = 'grzw1w2', validation_fraction = 0.3, test_fraction = 0.2, downsample_field_srcs = False, downsample_field_srcs_fraction = 2.0, drop_missing = True, random_state = 42):
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
        data_validation = photo_cat_validation[photo_cat_validation.x_ray_flux_bin == i]

        X_val = data_validation[features_cols]
        y_val = data_validation[target_col]
        


        data = photo_cat_train_test[photo_cat_train_test.x_ray_flux_bin == i]

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



def find_completeness_purity_intercept(cutoffs, completeness, purity):

    cutoff_intersection_id = np.argmin(np.abs(completeness[completeness>0] - purity[completeness>0]))
    cutoff_intersection = cutoffs[completeness>0][cutoff_intersection_id]
    completeness_intersection = completeness[completeness>0][cutoff_intersection_id]
    purity_intersection = purity[completeness>0][cutoff_intersection_id]

    return cutoff_intersection, completeness_intersection, purity_intersection

def assess_goodnes_nway_cross_match(nway_res_ero, plot_res = True):
    test_columns = ['EROSITA','ID', 'pos_err', 'Separation_EROSITA_DESI',  'prob_has_match', 'prob_this_match', 'match_flag', 'desi_id',  'desi_id_true_ctp']
    print("="*20)
    print('NWAY PERFOMANCE ON THE VALIDATION CATALOG')
    nway_res_ero = nway_res_ero.copy()
    test_df = nway_res_ero[~nway_res_ero.desi_id_true_ctp.isna()]

    test_df_matched = test_df.query('match_flag==1')

    #tmp_col = test_df_matched.desi_id == test_df_matched.desi_id_true_ctp
    test_df_matched['nway_equal_true'] = test_df_matched.desi_id == test_df_matched.desi_id_true_ctp
    #test_df_matched.loc[tmp_col.values, 'nway_equal_true'] = tmp_col.values

    cutoffs = np.linspace(0.01,0.99,100)

    def calcu_stats(test_df_matched, cutoffs):
        total_ctps = len(test_df_matched)
        completeness = []
        purity = []
        for p in cutoffs:
            cutoff_mask = test_df_matched.prob_has_match > p
            true_check_mask = test_df_matched.nway_equal_true

            n_assigned_ctps = len(test_df_matched[cutoff_mask])
            if n_assigned_ctps == 0:
                C = 0
                P = 0
            else:
                true_assignment = len(test_df_matched[cutoff_mask & true_check_mask])
                false_assingment = len(test_df_matched[cutoff_mask & ~true_check_mask])

                C = n_assigned_ctps / total_ctps

                P = true_assignment / n_assigned_ctps

            completeness.append(C)
            purity.append(P)
        completeness = np.array(completeness)
        purity = np.array(purity)
        return cutoffs, completeness, purity

    cutoffs, completeness, purity = calcu_stats(test_df_matched, cutoffs)

    print('Completeness and purity for nway matching \n'+ 'completeness = fraction of sources with prob_has_match > p \n' + 'purity = fraction of sources with prob_has_match > p and correct nway assignment')

    plt.figure(figsize=(8,5))
    plt.plot(cutoffs, completeness, label='completeness')
    plt.plot(cutoffs, purity, label='purity')

    cutoff_intersection, completeness_intersection, purity_intersection = find_completeness_purity_intercept(cutoffs, completeness, purity)

    frac_src_p_any_over = (nway_res_ero.prob_has_match > cutoff_intersection ).astype(int).mean()
    frac_src_p_any_over = np.round(frac_src_p_any_over*100, 2)


    plt.axvline(cutoff_intersection, color='k', ls='--', label=f'purity=completeness={completeness_intersection:.2g}%; \n {frac_src_p_any_over:.2g}% of sources have prob_has_match > {cutoff_intersection:.2g}')

    plt.legend()
    plt.ylim(0.5, 1.05)
    plt.xlabel('prob_has_match cutoff')
    plt.ylabel('completeness/purity')

    print(f" Completeness = {100*completeness_intersection:.2g}% \n Purity = {100*purity_intersection:.2g}% \n prob_has_match optimal cutoff =  {cutoff_intersection:.2g} \n Fraction of sources with prob_has_match > {cutoff_intersection:.2g} = {frac_src_p_any_over:.2g}%")


    cutoff_mask = test_df_matched.prob_has_match > cutoff_intersection
    true_check_mask = test_df_matched.nway_equal_true

    n_assigned_ctps = len(test_df_matched[cutoff_mask])

    true_assignment = len(test_df_matched[cutoff_mask & true_check_mask])
    false_assingment = len(test_df_matched[cutoff_mask & ~true_check_mask])

    print('+++Statistics+++')
    print(f"{len(test_df_matched)} X-ray sources in validation set with counterparts") 
    print(f"--Out of those, {len(test_df_matched)-n_assigned_ctps} sources were assigned hostless (prob_has_match < {cutoff_intersection:.2g}) ")
    print(f"{n_assigned_ctps} sources have prob_has_match > {cutoff_intersection:.2g}")
    print(f'Out of those {n_assigned_ctps}: ')
    print(f"--{true_assignment} sources have correct nway counterpart")
    print(f"--{false_assingment} sources have incorrect nway counterpart")


    if not plot_res:
        plt.close()

    return cutoff_intersection, completeness_intersection,  cutoffs, completeness, purity



def assess_goodnes_srgz_cross_match(srgz_res_ero, plot_res = True):
    print("="*20)
    print('SRGz PERFOMANCE ON THE VALIDATION CATALOG')
    srgz_res_ero = srgz_res_ero.copy()
    test_df_matched = srgz_res_ero[~srgz_res_ero.desi_id_true_ctp.isna()]

    test_df_matched.iloc[:, 'srgz_equal_true'] = test_df_matched.desi_id == test_df_matched.desi_id_true_ctp


    cutoffs = np.linspace(0.01,srgz_res_ero['P_0'].max,100)

    def calcu_stats(test_df_matched, cutoffs):
        total_ctps = len(test_df_matched)
        completeness = []
        purity = []
        for p in cutoffs:
            cutoff_mask = test_df_matched.prob_has_match > p
            true_check_mask = test_df_matched.srgz_equal_true

            n_assigned_ctps = len(test_df_matched[cutoff_mask])
            if n_assigned_ctps == 0:
                C = 0
                P = 0
            else:
                true_assignment = len(test_df_matched[cutoff_mask & true_check_mask])
                false_assingment = len(test_df_matched[cutoff_mask & ~true_check_mask])

                C = n_assigned_ctps / total_ctps

                P = true_assignment / n_assigned_ctps

            completeness.append(C)
            purity.append(P)
        completeness = np.array(completeness)
        purity = np.array(purity)
        return cutoffs, completeness, purity

    cutoffs, completeness, purity = calcu_stats(test_df_matched, cutoffs)

    print('Completeness and purity for srgz matching \n'+ 'completeness = fraction of sources with prob_has_match > p \n' + 'purity = fraction of sources with prob_has_match > p and correct srgz assignment')

    plt.figure(figsize=(8,5))
    plt.plot(cutoffs, completeness, label='completeness')
    plt.plot(cutoffs, purity, label='purity')

    cutoff_intersection, completeness_intersection, purity_intersection = find_completeness_purity_intercept(cutoffs, completeness, purity)

    frac_src_p_any_over = (srgz_res_ero.prob_has_match > cutoff_intersection ).astype(int).mean()
    frac_src_p_any_over = np.round(frac_src_p_any_over*100, 2)


    plt.axvline(cutoff_intersection, color='k', ls='--', label=f'purity=completeness={completeness_intersection:.2g}%; \n {frac_src_p_any_over:.2g}% of sources have prob_has_match > {cutoff_intersection:.2g}')

    plt.legend()
    plt.ylim(0.5, 1.05)
    plt.xlabel('prob_has_match cutoff')
    plt.ylabel('completeness/purity')

    print(f" Completeness = {100*completeness_intersection:.2g}% \n Purity = {100*purity_intersection:.2g}% \n prob_has_match optimal cutoff =  {cutoff_intersection:.2g} \n Fraction of sources with prob_has_match > {cutoff_intersection:.2g} = {frac_src_p_any_over:.2g}%")


    cutoff_mask = test_df_matched.prob_has_match > cutoff_intersection
    true_check_mask = test_df_matched.srgz_equal_true

    n_assigned_ctps = len(test_df_matched[cutoff_mask])

    true_assignment = len(test_df_matched[cutoff_mask & true_check_mask])
    false_assingment = len(test_df_matched[cutoff_mask & ~true_check_mask])

    print('+++Statistics+++')
    print(f"{len(test_df_matched)} X-ray sources in validation set with counterparts") 
    print(f"--Out of those, {len(test_df_matched)-n_assigned_ctps} sources were assigned hostless (prob_has_match < {cutoff_intersection:.2g}) ")
    print(f"{n_assigned_ctps} sources have prob_has_match > {cutoff_intersection:.2g}")
    print(f'Out of those {n_assigned_ctps}: ')
    print(f"--{true_assignment} sources have correct srgz counterpart")
    print(f"--{false_assingment} sources have incorrect srgz counterpart")


    if not plot_res:
        plt.close()

    return cutoff_intersection, completeness_intersection,  cutoffs, completeness, purity



def flux2mag(flux):

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
        pd.Series: _description_
    """

    flux = w_eff * ab_zeropoint * 10 ** (-0.4 * magnitudes)
    flux.name = 'flux_corrected'

    return flux


def desi_reliable_magnitudes(df: pd.DataFrame,
                        s_n_threshold: int = 4,
                        colors: bool=True,
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
    Returns:
        pd.DataFrame: Catalogue with reliable magnitudes.
    """
    df = df.copy()
    for band in ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:

        flux_colname = f'flux_{band}'
        flux_ivar_colname = f'flux_ivar_{band}'
        dered_mag_colname = f'dered_mag_{band}'


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

    W_EFF_Z = 1289.35
    AB_ZEROPOINT_Z = 1.29484e-9

    df['rel_desi_flux_corr_z'] = flux_frequency_correction(df['rel_dered_mag_z'],
                                                           w_eff=W_EFF_Z,
                                                           ab_zeropoint=AB_ZEROPOINT_Z)

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




    # if xray:
    #     # X-ray to optical flux
    #     df['lg(Fx/Fo_g)'] = np.log10(df['flux_05-20'] / df['flux_g'])
    #     df['lg(Fx/Fo_r)'] = np.log10(df['flux_05-20'] / df['flux_r'])
    #     df['lg(Fx/Fo_z)'] = np.log10(df['flux_05-20'] / df['flux_z'])

    #     '''
    #     TODO: update with datalab data when possible
    #     '''

    #     dered_flux_z = 10 ** (9 - df['rel_dered_mag_z'] / 2.5)
    #     df['rel_dered_lg(Fx/Fo_z)'] = np.log10(df['flux_05-20'] / dered_flux_z)
    #     df['rel_dered_lg(Fx/Fo_z_corr)'] = np.log10(df['flux_05-20'] / df['rel_desi_flux_corr_z'])

    #     dered_flux_g = 10 ** (9 - df['rel_dered_mag_g'] / 2.5)
    #     df['rel_dered_lg(Fx/Fo_g)'] = np.log10(df['flux_05-20'] / dered_flux_g)

    #     dered_flux_r = 10 ** (9 - df['rel_dered_mag_r'] / 2.5)
    #     df['rel_dered_lg(Fx/Fo_r)'] = np.log10(df['flux_05-20'] / dered_flux_r)

    return df



def rayleigh_plot(input_cross_match_df, sep_col = 'sep', pos_err_col = 'pos_err',
    pos_err_corr_func = lambda x: x, corr_error_str='err*1.0', plotlabel = 'eROSITAx',
    xlim=(0, 3), ylim=(1e-3, 1)): 
    '''
    input_cross_match_df - dataframe with cross-matched catalog 1 with catalog 2.
    all cuts and queries should be done before calling this function.
    '''

    pos_err = pos_err_corr_func(input_cross_match_df[pos_err_col])
    corrected_pos_err = pos_err

    rat = input_cross_match_df[sep_col]/corrected_pos_err

    rayleigh_fit = stats.rayleigh.fit(rat)
    #sns.histplot(ero_ctps_tmp, x = rat, bins=50, stat = 'density', ax = ax)

    fig, axs =  plt.subplots(nrows=2, ncols = 1, sharex = True, gridspec_kw = {'hspace':0, 'height_ratios': None}, figsize = (12,12))
    ax, ax2 = axs
    sns.ecdfplot(input_cross_match_df, x = rat, ax = ax, complementary = True, lw = 3)
    sns.histplot(input_cross_match_df, x = rat, ax = ax2, stat = 'density', lw = 3, bins = 50)

    for prob in [39.3, 68, 95, 98]:
        ax.axhline(1 - prob/100, color = 'k', ls = '--', alpha = 0.5)
        ax.text(0.5, 1 - prob/100, f'{prob}%', ha = 'center', va = 'center', color = 'k', alpha = 0.5)


    #plot the fit
    x = np.linspace(0, rat.max()*1.05, 100)
    ax.plot(x, 1-stats.rayleigh.cdf(x, *rayleigh_fit), 'r-', lw=3, alpha=0.6, label='Rayleigh fit: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % rayleigh_fit, zorder = -1)
    ax.plot(x, 1-stats.rayleigh.cdf(x, 0,1), 'g-', lw=3, alpha=0.6, label='Rayleigh fixed: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % (0,1), zorder = -1)
    ax.set(ylim=ylim, xlim=xlim)
    ax2.set_xlabel('Separation/corrected_pos_err; \n '+ 'corr_error='+corr_error_str)
    ax.set_yscale('log')

    ax2.plot(x, stats.rayleigh.pdf(x, *rayleigh_fit), 'r-', lw=3, alpha=0.6, label='Rayleigh fit: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % rayleigh_fit, zorder = -1)
    ax2.plot(x, stats.rayleigh.pdf(x, 0,1), 'g-', lw=3, alpha=0.6, label='Rayleigh fixed: '+'$\mu$ = %.2f, $\sigma$ = %.2f' % (0,1), zorder = -1)

    plt.legend()
    plt.suptitle(plotlabel+', '+str(len(input_cross_match_df))+' sources')




def cross_match_data_frames(df1: pd.DataFrame, df2: pd.DataFrame, 
                            colname_ra1: str, colname_dec1: str,
                            colname_ra2: str, colname_dec2: str,
                            match_radius: float = 3.0,
                            df_prefix: str = '',
                            closest: bool = False,
                            ):
    """
    cross_match_data_frames cross-matches two dataframes.
    Cross-match two dataframes with astropy
    https://docs.astropy.org/en/stable/api/astropy.coordinates.match_coordinates_sky.html#astropy.coordinates.match_coordinates_sky
    https://docs.astropy.org/en/stable/api/astropy.coordinates.search_around_sky.html#astropy.coordinates.search_around_sky
    Args:
        df1 (pd.DataFrame): first catalog
        df2 (pd.DataFrame): second catalog
        colname_ra1 (str): columns name for ra in df1
        colname_dec1 (str): columns name for dec in df1
        colname_ra2 (str): columns name for ra in df2
        colname_dec2 (str): columns name for dec in df2
        match_radius (float, optional): match radius in arcsec. Defaults to 3.0.
        df_prefix (str, optional): prefix to prepend to the columns of the second data frame. Defaults to ''.
        closest (bool, optional): whether to return the closest match. Defaults to False.

    Returns:
        pd.DataFrame: match of df1 and df2


    example:
    cross_match_data_frames(desi, gaia, 
                                colname_ra1='RA_fin',
                                colname_dec1='DEC_fin',
                                colname_ra2='ra',
                                colname_dec2='dec',
                                match_radius = 10,
                                df_prefix = 'GAIA',
                                closest=False)
    """
    df1 = df1.copy()
    orig_size = df1.shape[0]
    df2 = df2.copy()
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)

    coors1 = SkyCoord(df1[colname_ra1]*u.degree, df1[colname_dec1]*u.degree, frame='icrs')
    coors2 = SkyCoord(df2[colname_ra2]*u.degree, df2[colname_dec2]*u.degree, frame='icrs')

    idx1, idx2, ang_sep, _ = coordinates.search_around_sky(coors1, coors2, match_radius*u.arcsec)

    ang_sep = pd.DataFrame({df_prefix+'_sep': ang_sep})

    df1 = df1.loc[idx1]
    df1.reset_index(drop=True, inplace=True)
    df2 = df2.loc[idx2]
    df2.reset_index(drop=True, inplace=True)
    df2.columns  = [df_prefix+'_'+x for x in df2.columns]
    df_matched = pd.concat([df1,ang_sep, df2], axis=1) 
    df_matched.sort_values(by=['index', df_prefix+'_sep'], inplace=True, ascending=True)

    print('cross-match radius', match_radius, 'arcsec')
    print('total matches:', len(df_matched), 'out of', orig_size)

    if closest:
        df_matched = df_matched.drop_duplicates(subset=['index'], keep='first')
        print('total closest matches:', len(df_matched))

    df_matched.drop(columns=['index'], inplace=True)
    df_matched.drop(columns=[df_prefix+'_index'], inplace=True)

    return df_matched                  
