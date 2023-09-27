"""
created matt_dumont 
on: 27/09/23
"""
from get_n_data import get_all_n_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from project_base import unbacked_dir

plot_dir = unbacked_dir.joinpath('raw_n_plots')
plot_dir.mkdir(exist_ok=True)


def plot_all_data():
    ndata = get_all_n_data()

    for site in ndata['site_id'].unique():
        fig, ax = plt.subplots()
        site_type = ndata.loc[ndata['site_id'] == site,'type'].unique()[0]
        for outlier, c in zip([True, False], ['r', 'b']):
            site_data = ndata[(ndata['site_id'] == site) & (ndata['outlier_auto'] == outlier)]
            ax.scatter(site_data['datetime'], site_data['n_conc'], color=c, label='outlier' if outlier else 'inlier')
        ax.set_title(f'{site_type} {site}')
        ax.legend()
        fig.savefig(plot_dir.joinpath(f'{site}.png'))
        plt.close(fig)


if __name__ == '__main__':
    plot_all_data()
