"""
created matt_dumont 
on: 24/11/23
"""

# todo
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from solvers.DreamzsBPEFM import DreamzsBpefmSolver
from pydream.parameters import SampledParam
from scipy.stats import uniform

from site_selection.get_n_data import get_n_metadata, plot_single_site, get_all_n_data
import geopandas as gpd
from project_base import unbacked_dir

base_dir = unbacked_dir.joinpath('BASE')
base_dir.mkdir(exist_ok=True)

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
    'Selwyn River-Coes Ford mrt-10',
    'Selwyn River-Coes Ford mrt-30',
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
        fig, ax, handles, labels = plot_single_site(site, ndata, data)
        fig.savefig(plt_dir.joinpath(f'{site}.png'))


def get_dreamz(site, rerun=False):
    ndata = get_all_n_data()
    data = get_n_metadata()
    data = data.loc[site]
    ndata = ndata.loc[ndata.site==site]


    ninf = 20
    model_name = f'{site}_annual'

    dbs = DreamzsBpefmSolver(save_dir=base_dir, n_inflections='A', # todo maybe more frequent breakpoints for some sites
                             ts_data=conc_data,
                             inflection_times=None, cdf_inflection_start=0.05,
                             **age_kwargs)

    starts = [0.05] + [0.05] * (2) + [0.05] * (ninf - 1)  # todo could auto generate from ts_data???
    endval = [0.15] + [1.0] * (2) + [5.0] * (ninf - 1)
    params = [
        SampledParam(uniform, starts, endval),
    ]

    sampled_params, log_ps = dbs.run_dreamzs(
            model_name=model_name,
            params=params,
            niterations=10000, nchains=5, starts=None, verbose=True,
            nverbose=100, restart=False, start_random=True, hardboundaries=True, multitry=False,rerun=rerun)
    fig, ax = dbs.plot_best_params_pred(model_name=model_name, nplot=0.05)
    ax[0].set_xlim(pd.to_datetime('2000-01-01'), pd.to_datetime('2025-01-01'))
    fig.tight_layout()
    fig.savefig(outdir.joinpath(f'{model_name}_pred.png'))

    fig, ax = dbs.plot_best_params(model_name=model_name, nplot=0.10)
    ax.set_xlim(pd.to_datetime('2000-01-01'), pd.to_datetime('2025-01-01'))
    fig.tight_layout()
    fig.savefig(outdir.joinpath(f'{model_name}_params.png'))


if __name__ == '__main__':
    pass
