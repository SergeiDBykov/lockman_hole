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
import astropy.io.fits as fits
import healpy as hp

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
    map_resol_deg = np.rad2deg(hp.nside2resol(1024))
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






def pandas_to_fits(dataframe: pd.DataFrame,
                    filename: str,
                    table_header_name: str,
                    sky_area_deg2: float):
    """
    pandas_to_fits saves a pandas dataframe as a fits file with all columns. Saves to data_path + filename.fits
    ##https://github.com/JohannesBuchner/nway/blob/master/nway-write-header.py
    Args:
        dataframe (pd.DataFrame): dataframe to save
        filename (str): filename (with path and .fits extension)
        table_header_name (str): header of the table, e.g. eROSITA-LHC
        sky_area_deg2 (float): sky area of the survey (needed for NWAY)
    """
    table = Table.from_pandas(dataframe)
    table.write(data_path+'/'+filename, overwrite = True)

    #https://github.com/JohannesBuchner/nway/blob/master/nway-write-header.py
    with fits.open(data_path+'/'+filename , 'update') as file:
        file[1].name = table_header_name
        file[1].header['SKYAREA'] = sky_area_deg2
        file.flush()
    return None


def fits_to_pandas(filename: str,):
    data = Table.read(data_path+filename, format='fits')
    with fits.open(data_path+'/'+filename , 'readonly') as file:
        dataname = file[1].name
    dataframe = data.to_pandas()
    dataframe.reset_index(inplace=True)
    dataframe.rename(columns={'index': dataname}, inplace=True)

    return dataframe


def my_scaler_forward(df):
    df_scaled = df.copy()
    for col in df.columns:
        if col.startswith('mag_'):
            df_scaled[col] = df[col]/35
        elif col.startswith('col_'):
            df_scaled[col] = df[col]/10

    return df_scaled

def my_scaler_backward(df_scaled):
    df = df_scaled.copy()
    for col in df.columns:
        if col.startswith('mag_'):
            df[col] = df_scaled[col]*35
        elif col.startswith('col_'):
            df[col] = df_scaled[col]*10


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

    #recall is TP / (TP + FN) - We know it as purity -> number of true positives / number of all positives
    #precision is TP / (TP + FP) - We know it as completeness -> number of samples that are correctly classified out of all classified samples

    fig,  ax =  plt.subplots( figsize = (8,8))
    ax.plot(precision, recall, label=label, linewidth=2)
    ax.set_xlabel('Precision/completeness')
    ax.set_ylabel('Recall/putiry')
    plt.grid(True)
    ax.set_aspect('equal')
    id_optim = np.argmin(np.abs(precision-recall))
    threshold_optim = thresholds[id_optim]
    precision_optim = precision[id_optim]
    recall_optim = recall[id_optim]
    print('Optimal threshold: {:.2f}'.format(threshold_optim))
    print('Optimal precision: {:.2f}'.format(precision_optim))
    ax.plot(precision_optim, recall_optim, 'o', color='C5', label = f"({precision_optim:.2f},{recall_optim:.2f})")
    ax.plot([0,1], [0,1], '--', color='C4')

    cm = sklearn.metrics.confusion_matrix(y_test, predict_proba > threshold_optim)

    cm_str = f"TN: {cm[0,0]}  FP: {cm[0,1]}\nFN: {cm[1,0]}  TP: {cm[1,1]}"
    #add a string version of the confusion matrix, add true positive , false positive  etc labels
    



    ax.text(0.3, 0.15, cm_str, ha='center', va='center', transform=ax.transAxes)
    ax.legend()


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
        features_cols = ['mag_g','mag_r','mag_z','mag_w1','mag_w2', 'col_gr', 'col_rz',  'col_gz','col_zw1', 'col_rw2', 'col_w1w2']
    elif features_cols == 'grz':
        features_cols = ['mag_g','mag_r','mag_z', 'col_gr', 'col_rz', 'col_gz']
    elif features_cols == 'grzw1w2w3w4':
        features_cols = ['mag_g','mag_r','mag_z','mag_w1','mag_w2', 'mag_w3', 'mag_w4', 'col_gr', 'col_rz', 'col_gz','col_zw1', 'col_rw2', 'col_w1w2', 'col_zw3', 'col_rw4', 'col_w3w4']

    target_col = ['is_counterpart']
    photo_cat = photo_cat_scaled.copy()
    if drop_missing:
        photo_cat.dropna(subset = features_cols, how = 'any', inplace = True)
    else:
        pass
    #assign random number to x_ray_flux_bin for each source which is not a counterpart
    tmp_col = np.random.randint(0, x_ray_flux_bins_num, len(photo_cat))
    photo_cat['x_ray_flux_bin'] = tmp_col
    

    flux_bin_num, flux_bins = pd.qcut(photo_cat[photo_cat.is_counterpart].flux_csc_05_2,  x_ray_flux_bins_num, retbins = True, labels = False)
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
