"""
created matt_dumont 
on: 1/11/23
"""

from project_base import unbacked_dir, generated_data_dir
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection_power.detection_power_calcs import get_trend_detection_power, get_trend_detection_power_no_noise, \
    get_no_trend_detection_power_no_noise, get_no_trend_detection_power, samp_durs, samp_freqs, get_plateau_power, get_all_plateau_sites
from site_selection.get_n_data import get_n_metadata, get_final_sites
from generate_true_concentration.gen_true_slope_init_conc import plot_single_site_source_recept


def plot_single_site_detection_power(site):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14, 10), gridspec_kw={'width_ratios': (1, 0.1)}
                            )
    conc_ax = axs[0, 0]
    power_ax = axs[1, 0]
    conc_leg = axs[0, 1]
    power_leg = axs[1, 1]
    power_leg.axis('off')
    conc_leg.axis('off')
    fig, conc_ax, handles, labels = plot_single_site_source_recept(site, ax=conc_ax)
    for i in range(4):
        handles.pop(2)
        labels.pop(2)

    conc_ax.get_legend().remove()
    conc_leg.legend(handles, labels, loc='center left')

    temp = conc_ax.get_title()
    conc_ax.set_title('')
    conc_ax.set_ylabel('Concentration (mg/L)')

    if site in get_all_plateau_sites():
        detect_noisy = get_plateau_power()
        detect_not_noisy=None
    else:
        detect_noisy = pd.concat([get_no_trend_detection_power(), get_trend_detection_power()])
        detect_noisy = detect_noisy.loc[detect_noisy.site == site]

        detect_not_noisy = pd.concat([get_trend_detection_power_no_noise(), get_no_trend_detection_power_no_noise()])
        detect_not_noisy = detect_not_noisy.loc[detect_not_noisy.site == site]

    use_samp_durs = pd.to_datetime('2010-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')

    if detect_not_noisy is not None:
        # plot noise free detection power
        freq = max(samp_freqs)
        plt_data = [detect_not_noisy.loc[f'{site}_{d}_{freq}', 'power'] for d in samp_durs]

        power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'Noise free detection', c='k', alpha=0.5)

    # plot detection power
    colors = ['firebrick', 'orange', 'royalblue', 'purple']
    for i, (freq, c) in enumerate(zip(samp_freqs, colors)):
        plt_data = [detect_noisy.loc[f'{site}_{d}_{freq}', 'power'] for d in samp_durs]

        power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'{freq} samples/yr', c=c)

    power_ax.set_ylabel('Detection Power (%)')
    for v in np.arange(0, 110, 10):
        power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.set_ylim(-5, 105)
    for d in samp_durs:
        d = pd.to_datetime('2010-01-01') + pd.to_timedelta(d * 365.25, unit='day')
        power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
        conc_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.axvline(pd.to_datetime('2010-01-01'), ls=':')
    power_leg.legend(*power_ax.get_legend_handles_labels(), loc='center left')
    power_ax.set_xlim(pd.to_datetime('2000-01-01'), pd.to_datetime('2065-01-01'))
    fig.suptitle(temp)
    fig.supxlabel('Time (years)')
    return fig


def plot_single_site_no_noise_power(site):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14, 10), gridspec_kw={'width_ratios': (1, 0.1)}
                            )
    conc_ax = axs[0, 0]
    power_ax = axs[1, 0]
    conc_leg = axs[0, 1]
    power_leg = axs[1, 1]
    power_leg.axis('off')
    conc_leg.axis('off')
    fig, conc_ax, handles, labels = plot_single_site_source_recept(site, ax=conc_ax)
    for i in range(4):
        handles.pop(2)
        labels.pop(2)

    conc_ax.get_legend().remove()
    conc_leg.legend(handles, labels, loc='center left')

    temp = conc_ax.get_title()
    conc_ax.set_title('')
    conc_ax.set_ylabel('Concentration (mg/L)')

    detect_not_noisy = pd.concat([get_no_trend_detection_power_no_noise(), get_trend_detection_power_no_noise()])
    detect_not_noisy = detect_not_noisy.loc[detect_not_noisy.site == site]

    use_samp_durs = pd.to_datetime('2010-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')
    # plot detection power
    colors = ['firebrick', 'orange', 'royalblue', 'purple']
    for i, (freq, c) in enumerate(zip(samp_freqs, colors)):
        plt_data = [detect_not_noisy.loc[f'{site}_{d}_{freq}', 'power'] for d in samp_durs]

        power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'{freq} samples/yr', c=c)

    power_ax.set_ylabel('Detection Power (%)')
    for v in np.arange(0, 110, 10):
        power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.set_ylim(-5, 105)
    for d in samp_durs:
        d = pd.to_datetime('2010-01-01') + pd.to_timedelta(d * 365.25, unit='day')
        power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
        conc_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.axvline(pd.to_datetime('2010-01-01'), ls=':')
    power_leg.legend(*power_ax.get_legend_handles_labels(), loc='center left')
    power_ax.set_xlim(pd.to_datetime('2000-01-01'), pd.to_datetime('2065-01-01'))
    fig.suptitle('Noise Free\n' + temp)
    fig.supxlabel('Time (years)')
    return fig


def _plot_all_no_noise(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for site in get_final_sites():
        print(site)
        fig = plot_single_site_no_noise_power(site)
        fig.tight_layout()
        fig.savefig(outdir.joinpath(f'{site}.png'))
        plt.close()

def plot_all_plateau_sites(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    sites = get_all_plateau_sites()
    for site in sites:
        print(site)
        fig = plot_single_site_detection_power(site)
        fig.tight_layout()
        plt.show()
        fig.savefig(outdir.joinpath(f'{site}.png'))
        plt.close()

def plot_all(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for site in get_final_sites():
        print(site)
        fig = plot_single_site_detection_power(site)
        fig.tight_layout()
        plt.show()
        fig.savefig(outdir.joinpath(f'{site}.png'))
        plt.close()


# todo for wells --> plot percentage of sites with {10+, 25+, 50+ 75+ 90+} (one plot each?) chance of detecting change
#  vs time and colored by samp_freq; EXCLUDE all zero chance wells
# todo for streams detection power vs MRT assumed different plots for different sampling frequencies v time
# todo write up plataue sites


if __name__ == '__main__':
    plot_all(unbacked_dir.joinpath('power_calc_site_plots')) # todo re-run when updated
    plot_all_plateau_sites(unbacked_dir.joinpath('power_calc_plateau_sites')) # todo re-run and check
    pass
