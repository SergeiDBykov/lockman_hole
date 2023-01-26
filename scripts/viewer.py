from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits
from astropy import units as u
from PIL import Image
import requests
import PIL.ImageOps as pops
import pandas as pd
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

from pathlib import Path


#credits for the major code: M. Belvederskiy


def search_around(ra: float, dec: float, catalog: pd.DataFrame, search_r_sec: float):
    """
    Return coordinated of sources from `catalog` within `search_r_sec` of (`ra`, `dec`)

    Args:
        ra: RA of center of search region
        dec: Dec of center of search region
        catalog: Catalog of sources to search
        search_r_sec: Search radius in arcseconds

    Returns:
        list: List of arrays of (ra, dec)
    """
    
    scalarc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)

    d2d = scalarc.separation(catalog)
    catalogmsk = d2d.arcsec < search_r_sec

    idxcatalog = np.where(catalogmsk)[0]

    # Coordinates of the sources around the center
    # and coordinates of the center
    desi_nearby_coord = catalog[idxcatalog]
    
    desi_nearby_ra = desi_nearby_coord.ra.deg
    desi_nearby_dec = desi_nearby_coord.dec.deg
    
    # Drop coordinates of the center
    desi_nearby_ra = np.delete(
        desi_nearby_ra, np.where(desi_nearby_ra == ra)
        )
    desi_nearby_dec = np.delete(
        desi_nearby_dec, np.where(desi_nearby_dec == dec)
        )

    return desi_nearby_ra, desi_nearby_dec


def neigbour_df(
    ra: float, dec: float, df: pd.DataFrame,
    ra_cat_column: str, dec_cat_column: str,
    search_r_sec: float
    ):
    """
    Return dataframe of neigbours of (`ra`, `dec`) from `df`
    within `search_r_sec` of (`ra`, `dec`)

    Args:
        ra (float): RA of center of search region
        dec (float): Dec of center of search region
        df (pd.DataFrame): Dataframe of sources to search
        ra_cat_column (str): Name of column in `df` with RA of sources
        dec_cat_column (str): Name of column in `df` with Dec of sources
        search_r_sec (float): Search radius in arcseconds

    Returns:
        pd.DataFrame: Dataframe of neigbours of (`ra`, `dec`)
    """

    # DESI neighbors
    desi_coord = SkyCoord(
        ra=df[ra_cat_column].values*u.degree,
        dec=df[dec_cat_column].values*u.degree
        )

    desi_nearby_ra, desi_nearby_dec = search_around(
        ra=ra, dec=dec, catalog=desi_coord, search_r_sec=search_r_sec
        )

    desi_nearby_featuers = df[
        df[ra_cat_column].isin(desi_nearby_ra) &
        df[dec_cat_column].isin(desi_nearby_dec)
        ]

    return desi_nearby_featuers


def desi_image_cutout_for_nway(
    nway_df: pd.DataFrame,
    ero_name: str,
    csc_df: Optional[pd.DataFrame] = None,
    xmm_df: Optional[pd.DataFrame] = None,
    jpeg: bool = True,
    ):
    """
    Credits:  M. Belvederskiy 
    Return cutout of DESI image around (`ero_ra`, `ero_dec`)
    
    Obtaining Images and Raw Data:
    https://www.legacysurvey.org/dr9/description/

    """

    ero_field_df = nway_df[nway_df['srcname_fin'] == ero_name]
    ero_close_field_df = ero_field_df.query('nway_Separation_EROSITA_DESI < 1.5 * pos_r98')

    ero_ra = ero_field_df.iloc[0]['RA_fin']
    ero_dec = ero_field_df.iloc[0]['DEC_fin']
    ero_error = ero_field_df.iloc[0]['pos_r98']
    ero_p_any = ero_field_df.iloc[0]['nway_prob_has_match']

    fits_url = f'https://www.legacysurvey.org/viewer/fits-cutout.fits?ra={ero_ra}&dec={ero_dec}&layer=ls-dr9&pixscale=0.2&bands=grz'
    jpeg_url = f'https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ero_ra}&dec={ero_dec}&layer=ls-dr9&pixscale=0.2&bands=grz'

    hdu = fits.open(fits_url)[0]

    # Image dimensions
    wcs = WCS(hdu.header)[0, :, :]
    fits_image_data = hdu.data[0, :, :]
    search_r_sec = 30

    desi_nearby_df = ero_field_df[['desi_ra', 'desi_dec', 'desi_id']]


    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(projection=wcs)

    # Color map etc.
    ax.imshow(fits_image_data, cmap='gray_r', origin='lower')
    
    if jpeg:
        jpeg_image = pops.invert(Image.open(requests.get(jpeg_url, stream=True).raw))
        ax.imshow(jpeg_image)

    # Save limits before DESI sources are plotted
    xlim_frozen = ax.get_xlim()
    ylim_frozen = ax.get_ylim()

    # Circle for ERO source
    ero_r = SphericalCircle(
        (ero_ra * u.deg, ero_dec * u.deg), ero_error * u.arcsec,
        edgecolor='k', facecolor='none', lw=.5,
        transform=ax.get_transform('icrs')
        )
    ax.add_patch(ero_r)

    # Central dot
    ax.scatter(
        ero_ra, ero_dec,
        transform=ax.get_transform('icrs'),
        s=200, marker='x', color='k', label='ERO', linewidths=.5
        )

    # DESI sources
    ax.scatter(
        desi_nearby_df['desi_ra'], desi_nearby_df['desi_dec'],
        transform=ax.get_transform('icrs'),
        s=300, edgecolor='blue', facecolor='none',
        lw=.5
        )

    for _, row in ero_close_field_df.iterrows():
        p_i = row['nway_prob_this_match']
        if p_i > 0.01:
            ax.text(
                (1 + 1e-6) * row['desi_ra'],
                (1 + 7e-6) * row['desi_dec'],
                f'{p_i:.0%}',
                transform=ax.get_transform('icrs'),
                color='k', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.1)
                )

            ax.text(
                (1 + 3e-6) * row['desi_ra'],
                (1 - 9e-6) * row['desi_dec'],
                row['desi_objid'],
                transform=ax.get_transform('icrs'),
                color='k', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.1)
                )
    

    if csc_df is not None:
        csc_nearby_df = neigbour_df(
            ero_ra, ero_dec, csc_df, 'ra', 'dec', search_r_sec
            )
        #print('non secure sources of XMM/CSC are not SHOWN!')
            # Error circles for the CSC sources
        for _, row in csc_nearby_df.iterrows():


            is_secure = row['secure']
            if is_secure:
                edgecolor = 'r'
                label = 'CSC'
            else:
                edgecolor = 'orange'
                label = 'CSC (not secure)'

            ax.scatter(
                csc_df['ra'], csc_df['dec'],
                transform=ax.get_transform('icrs'),
                s=10, color=edgecolor, label=label
                )
        
            csc_r = SphericalCircle(
                (row['ra'] * u.deg,
                row['dec'] * u.deg),
                row['r_98'] * u.arcsec,
                edgecolor=edgecolor, facecolor='none', lw=1,
                transform=ax.get_transform('icrs'),
                ls = '-'
                )
            ax.add_patch(csc_r)
            
            csc_r_false = SphericalCircle(
                (row['ra'] * u.deg,
                row['dec'] * u.deg),
                1.47 * u.arcsec,
                edgecolor=edgecolor, facecolor='none', lw=1,
                transform=ax.get_transform('icrs'),
                ls = '--'
                )
            ax.add_patch(csc_r_false)
            

    if xmm_df is not None:
        xmm_nearby_df = neigbour_df(
            ero_ra, ero_dec, xmm_df, 'sc_ra', 'sc_dec', search_r_sec
            )

            # Error circles for the xmm sources
        for _, row in xmm_nearby_df.iterrows():

            is_secure = row['secure']
            if is_secure:
                edgecolor = 'g'
                label = 'XMM'
            else:
                edgecolor = 'yellowgreen'
                label = 'XMM (not secure)'

            ax.scatter(
                xmm_df['sc_ra'], xmm_df['sc_dec'],
                transform=ax.get_transform('icrs'),
                s=10, color=edgecolor, label=label
                )
        
            xmm_r = SphericalCircle(
                (row['sc_ra'] * u.deg,
                row['sc_dec'] * u.deg),
                row['r_98'] * u.arcsec,
                edgecolor=edgecolor, facecolor='none', lw=1,
                transform=ax.get_transform('icrs'),
                ls = '-'
                )
            ax.add_patch(xmm_r)
            
            xmm_r_false = SphericalCircle(
                (row['sc_ra'] * u.deg,
                row['sc_dec'] * u.deg),
                1.47 * u.arcsec,
                edgecolor=edgecolor, facecolor='none', lw=1,
                transform=ax.get_transform('icrs'),
                ls = '--'
                )
            ax.add_patch(xmm_r_false)



    ax.set_title(f'{ero_name}, $p_{{any}}$: {ero_p_any:.2f}', fontsize=20, y=1.1)
    print()
    print(f'{ero_name}')
    print()

    ax.set(xlim=xlim_frozen, ylim=ylim_frozen)
    lgnd = ax.legend(loc='upper right', fontsize=20)
    for handle in lgnd.legendHandles:
        handle._sizes = [70]

    # ICRS coordinates
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='gray', ls='dotted')
    overlay[0].set_axislabel(' ')
    overlay[1].set_axislabel(' ')
    ax.coords[0].set_axislabel('Right Ascension')
    ax.coords[1].set_axislabel('Declination')


    ero_field_df_for_table = ero_field_df.copy()
    ero_field_df_for_table = ero_field_df_for_table.query('nway_prob_this_match>0.01').sort_values('nway_prob_this_match', ascending=False)
    ero_field_df_for_table['is_true_ctp'] = ero_field_df_for_table['desi_id']==ero_field_df_for_table['desi_id_true']
    ero_field_df_for_table['is_true_ctp'] = ero_field_df_for_table['is_true_ctp'].replace({True: '(True)', False: ''})
    ero_field_df_for_table['desi_objid'] = ero_field_df_for_table['desi_id'].str.split('_', 2).str[-1]
    #add (True) to the desi_objid if it is a true match
    ero_field_df_for_table['desi_objid'] = ero_field_df_for_table['desi_objid'] + ero_field_df_for_table['is_true_ctp']
    ero_field_df_for_table = ero_field_df_for_table[['desi_objid', 'nway_Separation_EROSITA_DESI', 'nway_prob_this_match', 'nway_photometry_nnmag_grzw1w2_orig', 'nway_photometry_nnmag_grzw1_orig', 'nway_photometry_nnmag_grz_orig']]
    ero_field_df_for_table.columns = ['DESI ID', 'sep', '$p_i$', 'grzw1w2', 'grzw1', 'grz']
    ero_field_df_for_table['nnmags'] =  [' '.join("{:.2f}".format(x).replace('-99.00','-') for x in y) for y in map(tuple, ero_field_df_for_table[['grzw1w2', 'grzw1', 'grz']].values)]

    ero_field_df_for_table.drop(['grzw1w2', 'grzw1', 'grz'], axis=1, inplace=True)
    ero_field_df_for_table.drop(['nnmags'], axis=1, inplace=True)

    ero_field_df_for_table = ero_field_df_for_table.round(2)
    #colWidths=[0.15, 0.1, 0.1, 0.2]
    colWidths=[0.15, 0.1, 0.1]
    colWidths = [x+0.03 for x in colWidths]
    mpl_table = ax.table(cellText=ero_field_df_for_table.values,
                        colLabels=ero_field_df_for_table.columns,
                        colWidths=colWidths,
                        cellLoc = 'center',
                        rowLoc= 'center',
                        colLoc='center',
    loc='upper left')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(14)
    mpl_table.scale(1, 2)

    plt.show()
    return fig

