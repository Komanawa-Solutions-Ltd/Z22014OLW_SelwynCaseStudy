"""
created matt_dumont 
on: 24/11/23
"""

import tempfile
import traceback
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from solvers.DreamzsBPEFM import DreamzsBpefmSolver
from pydream.parameters import SampledParam
from scipy.stats import uniform
from run_managers.run_multiprocess import run_multiprocess
from generators.normal_path_change import NormalPath
from site_selection.get_n_data import get_n_metadata, plot_single_site, get_all_n_data
import geopandas as gpd
from project_base import unbacked_dir, project_dir

base_dir = unbacked_dir.joinpath('BASE')
base_dir.mkdir(exist_ok=True)
run_dir = base_dir.joinpath('runs_normpath')
run_dir.mkdir(exist_ok=True)
logdir = base_dir.joinpath('logs_normpath')
logdir.mkdir(exist_ok=True)
nchains = 5
sites = [
    'Harts Creek - Lower Lake Rd mrt-10',
    'Harts Creek - Lower Lake Rd mrt-30',
    'l35_0191',
    'l36_0317',
    'l36_0477',
    'l36_0871',
    'm36_0698',
    'm36_3683',
    'm36_4126',
]


def plot_sites():
    ndata = get_all_n_data()
    data = get_n_metadata()
    data = data.loc[sites]
    outdata = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['nztmx'], data['nztmy']))
    outdata['datetime_min'] = outdata['datetime_min'].dt.strftime('%Y-%m-%d')
    outdata['datetime_max'] = outdata['datetime_max'].dt.strftime('%Y-%m-%d')
    outdata.to_file(base_dir.joinpath('selwyn_sites.shp'))

    plt_dir = base_dir.joinpath('data_plots')
    plt_dir.mkdir(exist_ok=True)
    for site in sites:
        mrt = data.loc[site, 'age_mean']
        auto_ex = False
        if 'Harts' in site:
            auto_ex = True
        if mrt < 5:
            rolling = '20D'  # approximately monthly
        elif 'Selwyn' in site:
            rolling = '300D'
        elif mrt < 20:
            rolling = '60D'  # approximately half annually
        else:
            rolling = '120D'
        fig, ax, handles, labels = plot_single_site(site, ndata, data, plot_auto_exlcude=auto_ex)
        fig.savefig(plt_dir.joinpath(f'{site}.png'))


def get_dreamz(site, rerun=False):
    ndata = get_all_n_data()
    data = get_n_metadata()
    data = data.loc[site]
    if 'Hart' in site:

        idx = (ndata.site_id == site) & ~ndata.always_exclude & ~ndata.unsup_outlier_auto.astype(bool)
    else:
        idx = (ndata.site_id == site) & ~ndata.always_exclude

    ndata = ndata.loc[idx].set_index('datetime').sort_index()['n_conc']

    mrt = mrt_p1 = data['age_mean']
    f_p1 = data['f_p1']
    model_name = f'{site}_BASE'
    precision = 2
    if mrt < 5:
        break_freq = '60D'
        precision = 3
    elif mrt < 20:
        break_freq = '90D'
    else:
        break_freq = '180D'

    if site in ['l36_0317', 'm36_4126']:
        start_bounds = (2, 10)
    else:
        start_bounds = (0.1, 5)

    dbs = DreamzsBpefmSolver(save_dir=run_dir, n_inflections=break_freq,
                             ts_data=ndata,
                             inflection_times=None, cdf_inflection_start=0.05,
                             mrt=mrt, mrt_p1=mrt_p1, mrt_p2=None, frac_p1=1, f_p1=f_p1,
                             f_p2=0.5,  # dummy
                             precision=precision)

    params = [SampledParam(NormalPath, start_bounds, 0.1, 20, 1, dbs.n_inflections)]

    sampled_params, log_ps = dbs.run_dreamzs(
        model_name=model_name,
        params=params,
        niterations=10000, nchains=nchains, starts=None, verbose=True,
        nverbose=100, restart=False, start_random=True, hardboundaries=True, multitry=False, rerun=rerun,
    )

    return dbs, model_name


def plot_base(site, outdir):
    data = get_n_metadata().loc[site]
    mrt = data['age_mean']
    f_p1 = data['f_p1']
    depth = data['depth']
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    dbs, model_name = get_dreamz(site)
    fig, ax = dbs.plot_best_params_pred(model_name=model_name, nplot=0.2, sharey=False,
                                        title_additon=f'\n{depth=} m {mrt=} yr, {f_p1=}', percentiles=(5, 25),
                                        pdf_alpha=0.25, plot_annual_vlines=True, plot_inflection_points=False)
    fig.tight_layout()
    fig.savefig(outdir.joinpath(f'{model_name}_pred.png'))


def run_all(rerun=False):
    for site in sites:
        print(f'running {site}')
        dbs, model_name = get_dreamz(site, rerun=rerun)
        fig, ax = dbs.plot_best_params_pred(model_name=model_name, nplot=0.05)
        plt.show()


def _get_dreamz_mp(kwargs):
    try:
        get_dreamz(**kwargs)
    except Exception:
        t = traceback.format_exc()
        with logdir.joinpath(f'{kwargs["site"]}_{datetime.datetime.now().isoformat()}.log').open('w') as f:
            f.write(t)


def run_all_mp(rerun=False):
    runs = []
    for site in sites:
        runs.append(dict(site=site, rerun=rerun))
    run_multiprocess(_get_dreamz_mp, runs, subprocess_cores=nchains)
    for site in sites:
        print(f'plotting {site}')
        plot_base(site, outdir=project_dir.joinpath('BASE_plots_normpath'))


if __name__ == '__main__':
    run_all_mp()
    pass
