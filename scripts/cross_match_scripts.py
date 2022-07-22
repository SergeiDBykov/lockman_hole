from .utils import data_path, set_mpl
from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.table import Table 
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
import healpy as hp

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







def my_scaler_forward(df):
    df_scaled = df.copy()
    for col in ['mag_g',	'mag_r',	'mag_z',	'mag_w1',	'mag_w2' ]:
        df_scaled[col] = df[col]/35
    for col in ['col_gr',	'col_rz',	'col_zw1',	'col_rw2']:
        df_scaled[col] = df[col]/10
    return df_scaled

def my_scaler_backward(df_scaled):
    df = df_scaled.copy()
    for col in ['mag_g',	'mag_r',	'mag_z',	'mag_w1',	'mag_w2', ]:
        df[col] = df_scaled[col]*35
    for col in ['col_gr',	'col_rz',	'col_zw1',	'col_rw2']:
        df[col] = df_scaled[col]*10
    return df

