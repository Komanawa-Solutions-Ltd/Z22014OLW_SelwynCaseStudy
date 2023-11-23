"""
a quick figure for my hydrosoc preoentation

created matt_dumont 
on: 20/11/23
"""
import time
import warnings

from project_base import unbacked_dir, generated_data_dir
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection_power.detection_power_calcs import get_trend_detection_power, get_trend_detection_power_no_noise, \
    get_no_trend_detection_power_no_noise, get_no_trend_detection_power, samp_durs, samp_freqs, get_plateau_power, \
    get_all_plateau_sites
from site_selection.get_n_data import get_n_metadata, get_final_sites, sw_ages
from generate_true_concentration.gen_true_slope_init_conc import plot_single_site_source_recept, \
    get_site_true_recept_conc, get_site_true_recept_conc_no_change
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from site_selection.get_n_data import get_n_metadata, get_all_n_data, get_final_sites, plot_single_site, start_year, end_year
from project_base import generated_data_dir, precision, unbacked_dir
from gw_age_tools import predict_historical_source_conc, make_age_dist, check_age_inputs, predict_future_conc_bepm
from generate_true_concentration.gen_true_slope_init_conc import get_site_source_conc, reductions



def plot_variable_mrt(base_site, reduction=0.2):
    """
    Plot the affect of MRT on detection power
    :param base_site: base stream site
    :param reduction: fraction reduction
    :return:
    """
    assert reduction in reductions
    ndata = get_all_n_data()
    metadata = get_n_metadata()
    detect_noisy = pd.concat([get_no_trend_detection_power(), get_trend_detection_power()])
    use_samp_durs = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')
    nsamp = 12
    lw = 2

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14, 10),
                            gridspec_kw={'width_ratios': (1, 0.1), 'height_ratios': (2, 1)},
                            )
    conc_ax = axs[0, 0]
    power_ax = axs[1, 0]
    conc_leg = axs[0, 1]
    power_leg = axs[1, 1]
    power_leg.axis('off')
    conc_leg.axis('off')
    sites = [base_site + f' mrt-{age}' for age in sw_ages]
    fig, ax, handles, labels = plot_single_site(sites[0], ndata, metadata, ax=conc_ax, alpha=0.2)
    ax.get_legend().remove()
    ax.set_title('')
    handles = handles[1:]
    labels = labels[1:]
    for i in range(6):
        labels.pop(1)
        handles.pop(1)
    colors = ['r', 'darkorange', 'teal', 'fuchsia']
    for site, color in zip(sites, colors):
        label_site = site.split(" ")[-1]
        source = get_site_source_conc(site)
        source.loc[end_year-start_year] = source.loc[0] * (1 - reduction)
        source.loc[100] = source.loc[0] * (1 - reduction)
        source = source.loc[source.index >= -20].sort_index()
        receptor = get_site_true_recept_conc(site, reduction=reduction)
        t = ax.plot(
            pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(source.index.values * 365.25, unit='day'),
            source,
            color=color, ls='--', label=f'Source - {site}', lw=lw)
        handles.append(t[0])
        labels.append(f'{label_site}: Source')
        t = ax.plot(
            pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(receptor.index.values * 365.25, unit='day'),
            receptor, color=color, label=f'Modelled Receptor - {site}', lw=lw)
        handles.append(t[0])
        labels.append(f'{label_site}: Receptor, with red.')

        no_change_recept = get_site_true_recept_conc_no_change(site)
        t = ax.plot(pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(no_change_recept.index.values * 365.25, unit='day'),
                    no_change_recept, color=color, ls=':', lw=lw)
        handles.append(t[0])
        labels.append(f'{label_site}: Receptor, no red.')

        plt_data = [detect_noisy.loc[f'{site}_{d}_{nsamp}', 'power'] for d in samp_durs]
        power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'{label_site}: {nsamp} samples/yr', c=color)

    ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':', color='k', alpha=0.5)
    conc_leg.legend(handles, labels, loc='upper left')
    power_ax.set_ylabel('Detection Power (%)')
    for v in np.arange(0, 110, 10):
        power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.set_ylim(-5, 105)
    for d in samp_durs:
        d = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(d * 365.25, unit='day')
        power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
        conc_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':')
    power_leg.legend(*power_ax.get_legend_handles_labels(), loc='upper left')
    power_ax.set_xlim(pd.to_datetime('2000-01-01'), pd.to_datetime('2045-01-01'))
    fig.suptitle(f'Affect of MRT on detection power at {base_site} Reduction={reduction * 100}%')
    fig.supxlabel('Time (years)')
    fig.tight_layout()
    return fig

if __name__ == '__main__':
    fig = plot_variable_mrt('Harts Creek - Lower Lake Rd')
    fig.savefig(Path.home().joinpath('Downloads/mrt_matters.png'))