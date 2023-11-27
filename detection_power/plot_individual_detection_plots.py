"""
created matt_dumont 
on: 1/11/23
"""
import itertools

from project_base import unbacked_dir, generated_data_dir
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection_power.detection_power_calcs import get_trend_detection_power, get_trend_detection_power_no_noise, \
    get_no_trend_detection_power_no_noise, get_no_trend_detection_power, samp_durs, samp_freqs, get_plateau_power, \
    get_all_plateau_sites
from site_selection.get_n_data import get_n_metadata, get_final_sites, sw_ages, start_year, end_year
from generate_true_concentration.gen_true_slope_init_conc import plot_single_site_source_recept, \
    get_site_true_recept_conc, get_site_true_recept_conc_no_change, reductions


def plot_single_site_detection_power(site, reduction, plot_plateau_power=False):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14, 10), gridspec_kw={'width_ratios': (1, 0.1)}
                            )
    conc_ax = axs[0, 0]
    power_ax = axs[1, 0]
    conc_leg = axs[0, 1]
    power_leg = axs[1, 1]
    power_leg.axis('off')
    conc_leg.axis('off')
    fig, conc_ax, handles, labels = plot_single_site_source_recept(site, reduction=reduction, ax=conc_ax)
    for i in range(4):
        handles.pop(2)
        labels.pop(2)

    conc_ax.get_legend().remove()
    conc_leg.legend(handles, labels, loc='center left')

    temp = conc_ax.get_title()
    conc_ax.set_title('')
    conc_ax.set_ylabel('Concentration (mg/L)')

    if site in get_all_plateau_sites(reduction=reduction) and plot_plateau_power:
        detect_noisy = get_plateau_power(reduction=reduction)
        detect_not_noisy = None
    else:
        detect_noisy = pd.concat([get_no_trend_detection_power(), get_trend_detection_power()])
        detect_noisy = detect_noisy.loc[detect_noisy.site == site]

        detect_not_noisy = pd.concat([get_trend_detection_power_no_noise(), get_no_trend_detection_power_no_noise()])
        detect_not_noisy = detect_not_noisy.loc[detect_not_noisy.site == site]

    use_samp_durs = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')

    if detect_not_noisy is not None:
        # plot noise free detection power
        freq = max(samp_freqs)
        plt_data = [detect_not_noisy.loc[f'{site}_{d}_{freq}_{int(reduction * 100)}', 'power'] for d in samp_durs]

        power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'Noise free detection', c='k', alpha=0.5)

    # plot detection power
    colors = ['firebrick', 'orange', 'royalblue', 'purple']
    for i, (freq, c) in enumerate(zip(samp_freqs, colors)):
        plt_data = [detect_noisy.loc[f'{site}_{d}_{freq}_{int(reduction * 100)}', 'power'] for d in samp_durs]

        power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'{freq} samples/yr', c=c)

    power_ax.set_ylabel('Detection Power (%)')
    for v in np.arange(0, 110, 10):
        power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.set_ylim(-5, 105)
    for d in samp_durs:
        d = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(d * 365.25, unit='day')
        power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
        conc_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':')
    power_leg.legend(*power_ax.get_legend_handles_labels(), loc='center left')
    power_ax.set_xlim(pd.to_datetime('2000-01-01'), pd.to_datetime('2065-01-01'))
    fig.suptitle(temp)
    fig.supxlabel('Time (years)')
    return fig


def plot_single_site_no_noise_power(site, reduction):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14, 10), gridspec_kw={'width_ratios': (1, 0.1)}
                            )
    conc_ax = axs[0, 0]
    power_ax = axs[1, 0]
    conc_leg = axs[0, 1]
    power_leg = axs[1, 1]
    power_leg.axis('off')
    conc_leg.axis('off')
    fig, conc_ax, handles, labels = plot_single_site_source_recept(site, reduction=reduction, ax=conc_ax)
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

    use_samp_durs = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')
    # plot detection power
    colors = ['firebrick', 'orange', 'royalblue', 'purple']
    for i, (freq, c) in enumerate(zip(samp_freqs, colors)):
        plt_data = [detect_not_noisy.loc[f'{site}_{d}_{freq}_{int(reduction * 100)}', 'power'] for d in samp_durs]

        power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'{freq} samples/yr', c=c)

    power_ax.set_ylabel('Detection Power (%)')
    for v in np.arange(0, 110, 10):
        power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.set_ylim(-5, 105)
    for d in samp_durs:
        d = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(d * 365.25, unit='day')
        power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
        conc_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
    power_ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':')
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
        for red in reductions:
            fig = plot_single_site_no_noise_power(site, reduction=red)
            fig.tight_layout()
            fig.savefig(outdir.joinpath(f'{site}_red{int(red * 100)}.png'))
            plt.close()


def plot_all_plateau_sites(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for red in reductions:
        sites = get_all_plateau_sites(reduction=red)
        for site in sites:
            print(site)
            fig = plot_single_site_detection_power(site, plot_plateau_power=True, reduction=red)
            fig.tight_layout()
            fig.savefig(outdir.joinpath(f'{site}_red{int(red * 100)}.png'))
            plt.close()


def plot_all(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for site in get_final_sites():
        print(site)
        for red in reductions:
            fig = plot_single_site_detection_power(site, reduction=red)
            fig.tight_layout()
            fig.savefig(outdir.joinpath(f'{site}_red{int(red * 100)}.png'))
            plt.close()


def plot_stream_detection(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    metadata = get_n_metadata()

    sites = metadata.loc[metadata.type == 'stream'].index
    sites = sites[np.in1d(sites, get_final_sites())]
    sites = np.unique(['-'.join(e.split('-')[:-1]) for e in sites])
    detect_noisy = pd.concat([get_no_trend_detection_power(), get_trend_detection_power()])
    use_samp_durs = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')

    for site, red in itertools.product(sites, reductions):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(nrows=len(samp_freqs) + 1, ncols=2, width_ratios=(1, 0.1))
        power_axs = [fig.add_subplot(gs[i, 0]) for i in range(len(samp_freqs))]
        conc_ax = fig.add_subplot(gs[len(samp_freqs), 0])
        leg_ax = fig.add_subplot(gs[:, 1])
        leg_ax.axis('off')
        fig.suptitle(site.replace('mrt', '') + f' Detection Power: {int(red * 100)}% reduction')
        fig.supxlabel('Time')

        conc_ax.set_ylabel(f'Receptor Concentration\n(mg/l)')
        colors = ['firebrick', 'darkorange', 'darkcyan', 'indigo']
        for i, (power_ax, freq) in enumerate(zip(power_axs, samp_freqs)):
            power_ax.set_ylabel(f'{freq} samp. per year\nPower (%)')
            for mrt, c in zip(sw_ages, colors):
                if i == 0:
                    receptor = get_site_true_recept_conc(f'{site}-{mrt}', reduction=red)
                    t = conc_ax.plot(
                        pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(receptor.index.values * 365.25,
                                                                                unit='day'),
                        receptor, color=c, label=f'Mrt: {mrt} yr\nw/ reduct.')

                    no_change_recept = get_site_true_recept_conc_no_change(f'{site}-{mrt}')
                    t = conc_ax.plot(
                        pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(no_change_recept.index.values * 365.25,
                                                                                unit='day'),
                        no_change_recept, color=c, ls=':', label=f'MRT: {mrt} yr\n no reduct.')

                plt_data = [detect_noisy.loc[f'{site}-{mrt}_{d}_{freq}_{int(red * 100)}', 'power'] for d in samp_durs]

                power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'MRT: {mrt} yr', c=c)

            for v in np.arange(0, 110, 10):
                power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
            power_ax.set_ylim(-5, 105)
            for d in samp_durs:
                d = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(d * 365.25, unit='day')
                power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
                if i == 0:
                    conc_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
            power_ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':', label='reductions start')
            power_ax.set_xlim(pd.to_datetime('2005-01-01'), pd.to_datetime('2065-01-01'))
            if i == 0:
                conc_ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':')
                conc_ax.set_xlim(pd.to_datetime('2005-01-01'), pd.to_datetime('2065-01-01'))
            power_ax.set_xticklabels([])

        handles, labels = conc_ax.get_legend_handles_labels()
        h, l = power_axs[0].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        leg_ax.legend(handles, labels, loc='center left')
        fig.tight_layout()
        fig.savefig(outdir.joinpath(f'{site}_mrt_comp_red_{red}.png'))
        plt.close(fig)


def plot_well_overview_red(outpath, reduction):
    outpath = Path(outpath)
    outpath.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(nrows=len(samp_freqs), ncols=2, width_ratios=(1, 0.1))
    power_axs = [fig.add_subplot(gs[i, 0]) for i in range(len(samp_freqs))]
    leg_ax = fig.add_subplot(gs[:, 1])
    leg_ax.axis('off')
    fig.supxlabel('Time')
    fig.supylabel('Percent of groundwater network')
    metadata = get_n_metadata()
    detect_noisy = pd.concat([get_no_trend_detection_power(), get_trend_detection_power()])

    sites = metadata.loc[metadata.type != 'stream'].index
    sites = sites[np.in1d(sites, get_final_sites())]
    ngw_sites = len(sites)
    sites = sites[~np.in1d(sites, get_all_plateau_sites(reduction=reduction))]

    use_samp_durs = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')
    percent_limits = [25, 50, 80, 90]
    colors = ['firebrick', 'darkorange', 'darkcyan', 'indigo']

    fig.suptitle(f'Groundwater Detection Power for {int(reduction * 100)}% reduction\n'
                 f'Percent of sites which can show a reduction ({len(sites)}/{ngw_sites} sites)')
    for i, (power_ax, freq) in enumerate(zip(power_axs, samp_freqs)):
        power_ax.set_title(f'{freq} samp. per year')
        for pl, c in zip(percent_limits, colors):
            plt_data = []
            for d in samp_durs:
                idx = (
                        np.in1d(detect_noisy['site'], sites)
                        & (detect_noisy['samp_years'] == d)
                        & (detect_noisy['samp_per_year'] == freq)
                        & (detect_noisy['reduction'] == reduction)
                )
                assert idx.sum() == len(sites)
                idx = idx & (detect_noisy['power'] >= pl)
                plt_data.append(idx.sum() / len(sites) * 100)

            power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'Power ≥ {pl}% ', c=c)
            for v in np.arange(0, 110, 10):
                power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
            power_ax.set_ylim(-5, 105)
            for d in samp_durs:
                d = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(d * 365.25, unit='day')
                power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
        power_ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':', label='reductions start', color='k')
        power_ax.set_xlim(pd.to_datetime('2005-01-01'), pd.to_datetime('2065-01-01'))
    leg_ax.legend(*power_axs[0].get_legend_handles_labels(), loc='center left')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_well_overview_freq(outpath, freq, reductions=reductions):
    outpath = Path(outpath)
    outpath.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(nrows=len(reductions), ncols=2, width_ratios=(1, 0.1))
    power_axs = [fig.add_subplot(gs[i, 0]) for i in range(len(reductions))]
    leg_ax = fig.add_subplot(gs[:, 1])
    leg_ax.axis('off')
    fig.supxlabel('Time')
    fig.supylabel('Percent of groundwater network')
    for power_ax, reduction in zip(power_axs, reductions):
        metadata = get_n_metadata()
        detect_noisy = pd.concat([get_no_trend_detection_power(), get_trend_detection_power()])

        sites = metadata.loc[metadata.type != 'stream'].index
        sites = sites[np.in1d(sites, get_final_sites())]
        ngw_sites = len(sites)
        sites = sites[~np.in1d(sites, get_all_plateau_sites(reduction=reduction))]

        use_samp_durs = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(samp_durs * 365.25, unit='day')
        percent_limits = [25, 50, 80, 90]
        colors = ['firebrick', 'darkorange', 'darkcyan', 'indigo']

        fig.suptitle(f'Groundwater Detection Power for {freq} samples per year')
        power_ax.set_title(f'{int(reduction * 100)} % reduction\n'
                           f'Percent of sites which can show a reduction ({len(sites)}/{ngw_sites} sites)')
        for pl, c in zip(percent_limits, colors):
            plt_data = []
            for d in samp_durs:
                idx = (
                        np.in1d(detect_noisy['site'], sites)
                        & (detect_noisy['samp_years'] == d)
                        & (detect_noisy['samp_per_year'] == freq)
                        & (detect_noisy['reduction'] == reduction)
                )
                assert idx.sum() == len(sites)
                idx = idx & (detect_noisy['power'] >= pl)
                plt_data.append(idx.sum() / len(sites) * 100)

            power_ax.plot(use_samp_durs, plt_data, marker='o', label=f'Power ≥ {pl}% ', c=c)
            for v in np.arange(0, 110, 10):
                power_ax.axhline(v, color='k', ls=':', lw=0.5, alpha=0.3)
            power_ax.set_ylim(-5, 105)
            for d in samp_durs:
                d = pd.to_datetime(f'{start_year}-01-01') + pd.to_timedelta(d * 365.25, unit='day')
                power_ax.axvline(d, color='k', ls=':', lw=0.5, alpha=0.3)
        power_ax.axvline(pd.to_datetime(f'{start_year}-01-01'), ls=':', label='reductions start', color='k')
        power_ax.set_xlim(pd.to_datetime('2005-01-01'), pd.to_datetime('2065-01-01'))
    leg_ax.legend(*power_axs[0].get_legend_handles_labels(), loc='center left')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_all_overivew(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for red in reductions:
        plot_well_overview_red(outdir.joinpath(f'well_detection_overview_red{int(red * 100)}.png'), reduction=red)
    for freq in samp_freqs:
        plot_well_overview_freq(outdir.joinpath(f'well_detection_overview_freq{freq}.png'), freq=freq)


if __name__ == '__main__':
    rerun = False
    if rerun:
        plot_all_overivew(generated_data_dir.joinpath('overview_plots'))
        plot_all(generated_data_dir.joinpath('power_calc_site_plots'))
        plot_all_plateau_sites(generated_data_dir.joinpath('power_calc_plateau_sites'))
        plot_stream_detection(generated_data_dir.joinpath('power_mrt_comp'))
    pass
