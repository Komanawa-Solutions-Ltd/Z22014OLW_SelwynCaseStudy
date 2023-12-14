"""
created matt_dumont 
on: 14/12/23
"""
import itertools
from matplotlib import colormaps
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
from project_base import generated_data_dir, proj_root
from site_selection.get_n_data import get_n_metadata, get_final_sites
from detection_power.detection_power_calcs import get_all_plateau_sites, get_trend_detection_power, \
    get_no_trend_detection_power, reductions, samp_freqs, samp_durs, get_trend_detection_power_no_noise, \
    get_no_trend_detection_power_no_noise

base_map_path = proj_root.joinpath('original_data/topo250/nz-topo250-maps.jpg')
outdir = generated_data_dir.joinpath('geospatial_plots')
outdir.mkdir(exist_ok=True)

a4_port = np.array([8.3, 11.7])
a4_land = np.array([11.7, 8.3])


def plot_plateau_locs():
    final_sites = get_n_metadata().loc[get_final_sites()]
    fig, ax = plt.subplots(figsize=a4_land * 0.9)
    plt_background(ax)
    msize = 80

    colors = ['r', 'b', 'gold', 'fuchsia']
    idx = final_sites['type'] != 'stream'
    ax.scatter(final_sites.loc[idx, 'nztmx'], final_sites.loc[idx, 'nztmy'], c='k', label='Never', s=msize)
    for c, red in zip(colors, reductions):
        temp = final_sites.loc[get_all_plateau_sites(red)]
        temp = temp.loc[temp['type'] != 'stream']
        ax.scatter(temp['nztmx'], temp['nztmy'], c=c, label=f'$\leq${int(red * 100)}% reduction', s=msize)
    ax.legend(title='Plateau at:', loc='upper right')
    fig.tight_layout()
    fig.savefig(outdir.joinpath('plateau_locs.png'))
    plt.show()


def plot_detect_power_locs():
    final_sites = get_n_metadata().loc[get_final_sites()]
    final_sites = final_sites.loc[final_sites['type'] != 'stream']
    dp = pd.concat((get_trend_detection_power(), get_no_trend_detection_power()))
    dp_no_trend = pd.concat((get_trend_detection_power_no_noise(), get_no_trend_detection_power_no_noise()))
    colors = get_colors(samp_durs, cmap_name='plasma')
    for red, freq in itertools.product(reductions, samp_freqs):
        print(red, freq)
        fig = plt.figure(figsize=a4_port * 0.9)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 0.05])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
        cax = fig.add_subplot(gs[:, 1])
        cax.axis('off')
        plt_background(ax1)
        plt_background(ax2)
        msize = 80
        lmsize = 15
        handles = []
        labels = []
        ax1.scatter(final_sites.loc[:, 'nztmx'], final_sites.loc[:, 'nztmy'], color='k', label='Never', s=msize)
        ax2.scatter(final_sites.loc[:, 'nztmx'], final_sites.loc[:, 'nztmy'], color='k', label='Never', s=msize)
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=lmsize))
        labels.append('Never')
        for c, dur in zip(colors, reversed(samp_durs)):
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=lmsize))
            labels.append(f'{dur} yrs')
            idx = (
                    (dp.power >= 80)
                    & (dp.reduction == red)
                    & (dp.samp_per_year == freq)
                    & (dp.samp_years == dur)
                    & np.in1d(dp.site, final_sites.index)
            )
            temp_sites = dp.loc[idx, 'site']
            temp = final_sites.loc[temp_sites]
            ax1.scatter(temp['nztmx'], temp['nztmy'], color=c, label=f'{dur} years', s=msize)

            idx = (
                    (dp_no_trend.power >= 80)
                    & (dp_no_trend.reduction == red)
                    & (dp_no_trend.samp_per_year == freq)
                    & (dp_no_trend.samp_years == dur)
                    & np.in1d(dp_no_trend.site, final_sites.index)
            )
            temp_sites = dp_no_trend.loc[idx, 'site']
            temp = final_sites.loc[temp_sites]
            ax2.scatter(temp['nztmx'], temp['nztmy'], color=c, label=f'{dur} years', s=msize)

        cax.legend(handles, labels, loc='center left', title='years to detection\n($\geq$80% power)')
        ax1.set_title(f'Including noise, red.={int(red * 100)}%, freq.={freq}/yr')
        ax2.set_title(f'Lag only, red.={int(red * 100)}%')
        for ax in [ax1, ax2]:
            ax.set_ylim(5.137e6, 5.195e6)
            ax.set_xlim(1.505e6, 1.575e6)
        fig.tight_layout()
        fig.savefig(outdir.joinpath(f'detect_power_locs_red{int(red * 100)}_freq{freq}.png'))
        plt.close(fig)


def get_colors(vals, cmap_name='tab10'):
    n_scens = len(vals)
    if n_scens < 20:
        cmap = colormaps[cmap_name]
        colors = [cmap(e / (n_scens + 1)) for e in range(n_scens)]
    else:
        colors = []
        i = 0
        cmap = colormaps[cmap_name]
        for v in vals:
            colors.append(cmap(i / 20))
            i += 1
            if i == 20:
                i = 0
    return colors


def plt_background(ax):
    assert isinstance(ax, plt.Axes)
    ds = gdal.Open(str(base_map_path))
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    image = ds.ReadAsArray()
    if image.ndim == 3:  # if a rgb image then plot as greyscale
        image = image.mean(axis=0)
    ll = (minx, miny)
    ur = (maxx, maxy)

    ax.imshow(image, extent=[ll[0], ur[0], ll[1], ur[1]], cmap='gray')  #
    ax.set_xlim(1.505e6, 1.5880e6)
    ax.set_ylim(5.137e6, 5.195e6)


def plot_sites():
    metadata = get_n_metadata().loc[get_final_sites()]
    fig, ax = plt.subplots(figsize=a4_land * 0.9)
    plt_background(ax)
    ax.scatter(metadata['nztmx'], metadata['nztmy'], c='r')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_plateau_locs()
    plot_detect_power_locs()
