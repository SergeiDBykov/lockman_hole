import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from .utils import rep_path, data_path, set_mpl
from typing import Tuple, Optional

set_mpl()



def get_expo_map(exp_map_file: str = 'old_cat/exp_img.fits',
                 xcat: Optional[pd.DataFrame] = None,
                 plot: bool = True,
                 plot_star_idx: int = 0,):



    expo_fits = fits.open(data_path+exp_map_file)
    expo_map = expo_fits[0].data.T #Why? 
    wcs_expo = WCS(expo_fits[0].header)


    if xcat is not None:
        ##https://docs.astropy.org/en/stable/wcs/wcstools.html#matplotlib-plots-with-correct-wcs-projection

        cords = SkyCoord(xcat['RA_corr'], xcat['DEC_corr'], unit='deg', frame = 'fk5')
        coords_xcat = wcs_expo.world_to_pixel(cords)
    else:
        coords_xcat = None


    if plot:
        fig = plt.figure(figsize=(18,18))
        fig.add_subplot(111, projection=wcs_expo)
        plt.imshow(expo_map.T, origin='lower', cmap='PuBu')
        plt.colorbar()
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.grid()
        if xcat is not None:
            plt.scatter(coords_xcat[0], coords_xcat[1],   c = 'k', s=2, zorder = 10,)
            
            ra_plot, dec_plot= xcat.loc[plot_star_idx]['RA_corr'], xcat.loc[plot_star_idx]['DEC_corr']
            coords_tmp = SkyCoord(ra_plot, dec_plot, unit='deg', frame = 'fk5')
            pix_x_plot, pix_y_plot = wcs_expo.world_to_pixel(coords_tmp)
            plt.scatter([pix_x_plot], [pix_y_plot],   c = 'r',  s=30, zorder = 11,)

        plt.show()
    
    return expo_fits, expo_map, wcs_expo, coords_xcat


def lnls_xy(f):
   """
   For an arbitrary input array f compute 2 arrays x,y
   for plotting cumulative logN-logS-like curve, making a step on each vallue of f
   array f does not have to be sorted
   returns tuple (x,y)
   for plotting use plt.plot(*lnls_xy(fx))
   """
   zz=np.sort(f)[::-1]
   n=np.arange(0,len(zz),1)

   x=np.array(np.transpose([zz,zz])).flatten()
   y=np.array(np.transpose([n,n+1])).flatten()

   return x,y
