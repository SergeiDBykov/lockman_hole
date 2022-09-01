from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits
from astropy import units as u
from PIL import Image
import requests
import PIL.ImageOps as pops
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

from pathlib import Path

from astroquery.simbad import Simbad
customSimbad = Simbad()
# https://github.com/astropy/astroquery/blob/main/astroquery/simbad/data/votable_fields_dict.json
customSimbad.add_votable_fields(
    'distance_result', 'ra(d)', 'dec(d)',
    'otype', 'otype(V)', 'otype(S)', 'plx',
    'pmra', 'pmdec', 'plx_error'
    )

def search_around(ra, dec, catalog, search_r_sec):
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
    ra, dec, df,
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


def desi_image_cutout(
    ero_df: pd.DataFrame,
    ero_name: str,
    desi_df: pd.DataFrame,
    csc_df: pd.DataFrame = None,
    csc_ra_name: str = 'ra',
    csc_dec_name: str = 'dec',
    csc_error_name: str = 'r_98_csc',
    xmm_df: pd.DataFrame = None,
    xmm_ra_name: str = 'SC_RA',
    xmm_dec_name: str = 'SC_DEC',
    xmm_error_name: str = 'xmm_pos_r98',
    sdss_df: pd.DataFrame = None,
    sdss_ra_name: str = 'ra',
    sdss_dec_name: str = 'dec',
    jpeg: bool = True,
    description: str = '',
    save_path: str = ''
    ):
    """
    Return cutout of DESI image around (`ero_ra`, `ero_dec`)
    
    Obtaining Images and Raw Data:
    https://www.legacysurvey.org/dr9/description/


    Args:
        ero_ra: RA of the ERO source (center of the image)
        ero_dec: Dec of the ERO source
        ero_error: Error of the ERO source
        desi_df: Dataframe with DESI sources
        csc_df: Dataframe with CSC sources
        csc_ra_name: Name of the column with RA of the CSC sources
        csc_dec_name: Name of the column with Dec of the CSC sources
        csc_error_name: Name of the column with error of the CSC sources
        jpeg: If True, the image is saved as a JPEG file
        title: Title of the image
        description: Description of the image (adds to the image title)
        save_path: Path to save the image
    """

    ero_field_df = ero_df[ero_df['srcname_fin'] == ero_name]
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
    search_r_sec = 40

    desi_nearby_df = neigbour_df(ero_ra, ero_dec, desi_df, 'ra', 'dec', search_r_sec)
    desi_nearby_extended = desi_nearby_df.query('type != "PSF"')

    # Features of the nearby CSC sources
    csc_nearby_df = neigbour_df(
        ero_ra, ero_dec, csc_df, csc_ra_name, csc_dec_name, search_r_sec
        )
    # Features of the nearby XMM sources
    xmm_nearby_df = neigbour_df(
        ero_ra, ero_dec,
        xmm_df,
        ra_cat_column=xmm_ra_name,
        dec_cat_column=xmm_dec_name,
        search_r_sec=search_r_sec
        )
    # Features of the nearby SDSS sources
    sdss_nearby_df = neigbour_df(
        ero_ra, ero_dec, sdss_df, sdss_ra_name, sdss_dec_name, search_r_sec
    )

    # Simbad query
    simbad_search_r = 2 * ero_error
    coords = SkyCoord(ero_ra, ero_dec, unit='deg')
    simbad_table = customSimbad.query_region(coords, radius=simbad_search_r * u.arcsec)
    
    fig = plt.figure(figsize=(10, 10))
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
        desi_nearby_df['ra'], desi_nearby_df['dec'],
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
    
    # Extended DESI sources
    ax.scatter(
        desi_nearby_extended['ra'], desi_nearby_extended['dec'],
        transform=ax.get_transform('icrs'), marker='D',
        s=150, edgecolor='blue', facecolor='none',
        label='DESI extended', linewidth=.5
        )

    if len(csc_nearby_df) > 0:
        ax.scatter(
            csc_df[csc_ra_name], csc_df[csc_dec_name],
            transform=ax.get_transform('icrs'),
            s=10, color='r', label='CSC'
            )
        
        # Error circles for the CSC sources
        for _, row in csc_nearby_df.iterrows():
            csc_r = SphericalCircle(
                (row[csc_ra_name] * u.deg,
                row[csc_dec_name] * u.deg),
                row[csc_error_name] * u.arcsec,
                edgecolor='red', facecolor='none', lw=1,
                transform=ax.get_transform('icrs')
                )
            ax.add_patch(csc_r)
            
            csc_r_false = SphericalCircle(
                (row[csc_ra_name] * u.deg,
                row[csc_dec_name] * u.deg),
                1.43 * u.arcsec,
                edgecolor='red', facecolor='none', lw=1,
                transform=ax.get_transform('icrs'),
                ls='--'
                )
            ax.add_patch(csc_r_false)

    if len(xmm_nearby_df) > 0:
        ax.scatter(
            xmm_nearby_df[xmm_ra_name], xmm_nearby_df[xmm_dec_name],
            transform=ax.get_transform('icrs'),
            s=10, color='green', label='XMM'
            )

        # Error circles for the XMM sources
        for _, row in xmm_nearby_df.iterrows():
            xmm_r = SphericalCircle(
                (row[xmm_ra_name] * u.deg,
                row[xmm_dec_name] * u.deg),
                row[xmm_error_name] * u.arcsec,
                edgecolor='green', facecolor='none', lw=1,
                transform=ax.get_transform('icrs')
                )
            ax.add_patch(xmm_r)
            
            xmm_r_false = SphericalCircle(
                (row[xmm_ra_name] * u.deg,
                row[xmm_dec_name] * u.deg),
                1.43 * u.arcsec,
                edgecolor='green', facecolor='none', lw=1,
                transform=ax.get_transform('icrs'),
                ls='--'
                )
            ax.add_patch(xmm_r_false)

    if len(sdss_nearby_df) > 0:
        ax.scatter(
            sdss_nearby_df[sdss_ra_name], sdss_nearby_df[sdss_dec_name],
            transform=ax.get_transform('icrs'), marker='s',
            s=150, color='none', edgecolor='darkorange', label='SDSS'
            )

        for _, row in sdss_nearby_df.iterrows():
            ax.text(
                (1 + 1.2e-5) * row[sdss_ra_name],
                (1 - 2e-6) * row[sdss_dec_name],
                row['class'][:3],
                transform=ax.get_transform('icrs'), fontsize=12,
                color='k',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='darkorange', linewidth=2)
                )

    if simbad_table is not None:
        simbad_df = simbad_table.to_pandas()
        print(f'{len(simbad_df)} Simbad objects are found in {simbad_search_r:.1f}"')
        # print(simbad_df.head())

        ax.scatter(
            simbad_df['RA_d'], simbad_df['DEC_d'],
            transform=ax.get_transform('icrs'),
            s=30, color='none', edgecolor='lime', label='Simbad'
            )

        for _, row in simbad_df.iterrows():

            ax.text(
                (1 - 3e-6) * row['RA_d'],
                (1 - 1e-6) * row['DEC_d'],
                row['OTYPE'],
                transform=ax.get_transform('icrs'), fontsize=12,
                color='k',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='lime', linewidth=2)
                )

    # fig.suptitle(f'', y=1.03)
    ax.set_title(f'{ero_name}, p any: {ero_p_any:.0%}{description}', y=1.1)
    print()
    print(f'{ero_name}')
    print()

    ax.set(xlim=xlim_frozen, ylim=ylim_frozen)
    lgnd = ax.legend(loc='upper right')
    for handle in lgnd.legendHandles:
        handle._sizes = [70]

    # ICRS coordinates
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='gray', ls='dotted')
    overlay[0].set_axislabel('Right Ascension')
    overlay[1].set_axislabel('Declination')

    ero_field_df_for_table = ero_field_df.copy()
    ero_field_df_for_table = ero_field_df_for_table.query('nway_prob_this_match>0.01').sort_values('nway_prob_this_match', ascending=False)
    ero_field_df_for_table['is_true_ctp'] = ero_field_df_for_table['nway_desi_id']==ero_field_df_for_table['nway_desi_id_true_ctp']
    ero_field_df_for_table['is_true_ctp'] = ero_field_df_for_table['is_true_ctp'].replace({True: '(True)', False: ''})
    ero_field_df_for_table['desi_objid'] = ero_field_df_for_table['nway_desi_id'].str.split('_', 2).str[-1]
    #add (True) to the desi_objid if it is a true match
    ero_field_df_for_table['desi_objid'] = ero_field_df_for_table['desi_objid'] + ero_field_df_for_table['is_true_ctp']
    ero_field_df_for_table = ero_field_df_for_table[['desi_objid', 'nway_Separation_EROSITA_DESI', 'nway_prob_this_match', 'nway_nnmag_grzw1w2', 'nway_nnmag_grzw1', 'nway_nnmag_grz']]
    ero_field_df_for_table.columns = ['ID', 'sep', 'p_i', 'grzw1w2', 'grzw1', 'grz']
    ero_field_df_for_table['nnmags'] =  [' '.join("{:.2f}".format(x).replace('-99.00','-') for x in y) for y in map(tuple, ero_field_df_for_table[['grzw1w2', 'grzw1', 'grz']].values)]

    ero_field_df_for_table.drop(['grzw1w2', 'grzw1', 'grz'], axis=1, inplace=True)

    ero_field_df_for_table = ero_field_df_for_table.round(2)
    colWidths=[0.15, 0.1, 0.1, 0.2]
    mpl_table = ax.table(cellText=ero_field_df_for_table.values,
                        colLabels=ero_field_df_for_table.columns,
                        colWidths=colWidths,
                        cellLoc = 'center',
                        rowLoc= 'center',
                        colLoc='center',
    loc='upper left')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.scale(1, 2)  

    if save_path != '':
        plt.close()
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{save_path}{ero_name}.png', dpi=200, bbox_inches='tight')

    plt.show()
