"""
created matt_dumont 
on: 27/10/23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from site_selection.get_n_data import get_n_metadata, get_all_n_data, get_final_sites, plot_single_site, start_year, \
    end_year
from project_base import generated_data_dir, precision, unbacked_dir
from gw_age_tools import predict_historical_source_conc, make_age_dist, check_age_inputs, predict_future_conc_bepm

reductions = (0.05, 0.10, 0.20, 0.30)


def get_site_source_conc(site, recalc=False):
    save_path = generated_data_dir.joinpath('true_source_conc_slope_init.hdf')
    if save_path.exists() and not recalc:
        try:
            return pd.read_hdf(save_path, key=site)
        except KeyError:
            pass
    metadata = get_n_metadata().loc[site]
    init_conc = metadata[f'conc_{start_year}']
    mrt = metadata['age_mean']
    mrt_p1 = metadata['age_mean']
    mrt_p2 = metadata['age_mean']
    frac_p1 = metadata['frac_1']
    assert np.isclose(frac_p1, 1)
    f_p1 = metadata['f_p1']
    f_p2 = metadata['f_p1']
    prev_slope = metadata['slope_yr']
    min_conc = min(metadata['nmin'], 0.5)
    max_conc = max(metadata['nmax'], 20)

    if site in ['l36_1313', 'l36_2094', 'l36_2122']:
        min_conc = 0.01

        if site == 'l36_1313':
            p0 = [0.1, 7.0]
        elif site == 'l36_2094':
            p0 = [0.075, 10.0]
        elif site == 'l36_2122':
            p0 = [0.05, 10.0]
    else:
        p0 = None

    if np.isclose(prev_slope, 0):
        mrt, mrt_p2 = check_age_inputs(mrt, mrt_p1, mrt_p2, frac_p1, precision, f_p1, f_p2)
        age_step, ages, age_fractions = make_age_dist(mrt, mrt_p1, mrt_p2, frac_p1, precision, f_p1, f_p2)
        hist = pd.Series(index=ages * -1, data=init_conc)
    else:
        hist = predict_historical_source_conc(init_conc, mrt, mrt_p1, mrt_p2, frac_p1, f_p1, f_p2, prev_slope, max_conc,
                                              min_conc, start_age=np.nan, precision=precision, p0=p0)
        hist.to_hdf(save_path, key=site, complib='zlib', complevel=9)
    return hist


def get_site_true_recept_conc(site, reduction, recalc=False):
    save_path = generated_data_dir.joinpath('true_receptor_conc_slope_init.hdf')
    key = f'{site}_{int(reduction * 100)}'
    if save_path.exists() and not recalc:
        try:
            return pd.read_hdf(save_path, key=key)
        except KeyError:
            pass
    hist = get_site_source_conc(site)
    metadata = get_n_metadata().loc[site]
    init_conc = metadata[f'conc_{start_year}']
    mrt = metadata['age_mean']
    mrt_p1 = metadata['age_mean']
    mrt_p2 = metadata['age_mean']
    frac_p1 = metadata['frac_1']
    assert np.isclose(frac_p1, 1)
    f_p1 = metadata['f_p1']
    f_p2 = metadata['f_p1']
    prev_slope = metadata['slope_yr']
    min_conc = min(metadata['nmin'], 1)
    max_conc = max(metadata['nmax'], 20)
    hist.loc[end_year - start_year] = hist.loc[0] * (1 - reduction)
    hist.loc[100] = hist.loc[0] * (1 - reduction)
    receptor_conc = predict_future_conc_bepm(once_and_future_source_conc=hist,
                                             predict_start=-30, predict_stop=100,
                                             mrt_p1=mrt_p1, frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2, mrt=mrt, mrt_p2=None,
                                             fill_value=min_conc,
                                             fill_threshold=.5, precision=precision, pred_step=10 ** -precision)
    receptor_conc.to_hdf(save_path, key=site, complib='zlib', complevel=9)


def get_site_true_recept_conc_no_change(site, recalc=False):
    save_path = generated_data_dir.joinpath('true_receptor_conc_slope_init_no_change.hdf')
    if save_path.exists() and not recalc:
        try:
            return pd.read_hdf(save_path, key=site)
        except KeyError:
            pass
    hist = get_site_source_conc(site)
    metadata = get_n_metadata().loc[site]
    init_conc = metadata[f'conc_{start_year}']
    mrt = metadata['age_mean']
    mrt_p1 = metadata['age_mean']
    mrt_p2 = metadata['age_mean']
    frac_p1 = metadata['frac_1']
    assert np.isclose(frac_p1, 1)
    f_p1 = metadata['f_p1']
    f_p2 = metadata['f_p1']
    prev_slope = metadata['slope_yr']
    min_conc = min(metadata['nmin'], 1)
    max_conc = max(metadata['nmax'], 20)
    hist.loc[end_year - start_year] = hist.loc[0]
    hist.loc[100] = hist.loc[0]
    receptor_conc = predict_future_conc_bepm(once_and_future_source_conc=hist,
                                             predict_start=-30, predict_stop=100,
                                             mrt_p1=mrt_p1, frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2, mrt=mrt, mrt_p2=None,
                                             fill_value=min_conc,
                                             fill_threshold=.5, precision=precision, pred_step=10 ** -precision)
    receptor_conc.to_hdf(save_path, key=site, complib='zlib', complevel=9)
    return receptor_conc


def recalc_all_sites(recalc=False):
    """
    a convinence function to recalculate all sites
    :param recalc: boolean whether to recalculate all sites (True), or only those missing data (default)
    :return:
    """
    sites = get_final_sites()
    for i, site in enumerate(sites):
        print(f'{i + 1}/{len(sites)}: {site}')
        for red in reductions:
            get_site_source_conc(site, recalc=recalc)
            get_site_true_recept_conc(site, reduction=red, recalc=recalc)


def plot_single_site_source_recept(site, reduction, ax=None):
    ndata = get_all_n_data()
    metadata = get_n_metadata()
    fig, ax, handles, labels = plot_single_site(site, ndata, metadata, ax=ax)
    source = get_site_source_conc(site)
    source.loc[end_year - start_year] = source.loc[0] * (1 - reduction)
    source.loc[100] = source.loc[0] * (1 - reduction)
    source = source.loc[source.index >= -20].sort_index()
    receptor = get_site_true_recept_conc(site, reduction=reduction)
    t = ax.plot(
        pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(source.index.values * 365.25, unit='day'),
        source,
        color='gold', ls='--', label='Source')
    handles.append(t[0])
    labels.append('Source')
    t = ax.plot(
        pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(receptor.index.values * 365.25, unit='day'),
        receptor, color='orange', label='Modelled Receptor')
    handles.append(t[0])
    labels.append('Receptor, with reduction')

    no_change_recept = get_site_true_recept_conc_no_change(site)
    t = ax.plot(
        pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(no_change_recept.index.values * 365.25, unit='day'),
        no_change_recept, color='fuchsia', ls=':', label='Receptor, no reduction')
    handles.append(t[0])
    labels.append('Receptor, no reduction')

    ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':', color='k', alpha=0.5)
    ax.legend(handles, labels, loc='upper right')
    return fig, ax, handles, labels


def plot_all_sites():
    save_dir = unbacked_dir.joinpath('true_conc_plots')
    save_dir.mkdir(exist_ok=True)
    for site in get_final_sites():
        for red in reductions:
            fig, ax, handles, labels = plot_single_site_source_recept(site, reduction=red)
            fig.tight_layout()
            fig.savefig(save_dir.joinpath(f'{site}_red{int(red * 100)}_true_conc.png'))
            plt.close(fig)


def plot_fix_sites():
    # fixed
    sites = ['l36_0089', 'l36_1313', 'l36_2094', 'l36_2122', 'm36_0297']
    for site in sites:
        for red in reductions:
            fig, ax, handles, labels = plot_single_site_source_recept(site, reduction=red)
            fig.tight_layout()
        plt.show()


def recalc_problem_sites():
    # fixed
    sites = ['l36_1313', 'l36_2094', 'l36_2122']
    for site in sites:
        print(site)
        for red in reductions:
            get_site_source_conc(site, recalc=True)
            get_site_true_recept_conc(site, reduction=red, recalc=True)


if __name__ == '__main__':
    # recalc_all_sites(True)
    plot_all_sites()
