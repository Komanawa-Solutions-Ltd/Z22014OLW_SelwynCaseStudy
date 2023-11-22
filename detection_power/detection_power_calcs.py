"""
created matt_dumont 
on: 30/10/23
"""
import itertools

from project_base import generated_data_dir, unbacked_dir
from gw_detect_power import DetectionPowerCalculator
import numpy as np
import pandas as pd
from site_selection.get_n_data import get_n_metadata, get_final_sites, start_year
from generate_true_concentration.gen_true_slope_init_conc import get_site_true_recept_conc, reductions

samp_durs = np.arange(5, 55, 5)
samp_freqs = (1, 4, 12, 52)


def make_trend_meta_data():
    use_sites = get_final_sites()
    metadata = get_n_metadata()
    use_sites = use_sites[metadata.loc[use_sites, 'slope_yr'] > 0]

    out_use_conc = []
    idvs = []
    site_names = []
    error_vals = []
    samp_years_vals = []
    samp_per_year_vals = []
    initial_conc_vals = []
    red_vals = []
    previous_slope_vals = []
    mrt_vals = []
    mrt_p1_vals = []
    frac_p1_vals = []
    f_p1_vals = []
    for samp_dur, samp_freq, red in itertools.product(samp_durs, samp_freqs, reductions):
        site_names.extend(use_sites)
        idvs.extend(pd.Series(use_sites) + f'_{samp_dur}_{samp_freq}')
        error_vals.extend(metadata.loc[use_sites, 'noise'])
        samp_years_vals.extend([samp_dur] * len(use_sites))
        samp_per_year_vals.extend([samp_freq] * len(use_sites))
        initial_conc_vals.extend(metadata.loc[use_sites, f'conc_{start_year}'])
        red_vals.extend([red] * len(use_sites))
        previous_slope_vals.extend(metadata.loc[use_sites, 'slope_yr'])
        mrt_vals.extend(metadata.loc[use_sites, 'age_mean'])
        mrt_p1_vals.extend(metadata.loc[use_sites, 'age_mean'])
        frac_p1_vals.extend(metadata.loc[use_sites, 'frac_1'])
        f_p1_vals.extend(metadata.loc[use_sites, 'f_p1'])

        # get true conc
        for site in use_sites:
            temp = make_true_conc(site, samp_freq=samp_freq, samp_dur=samp_dur, reduction=red)
            out_use_conc.append(temp)

    outdata = pd.DataFrame(index=idvs, data={'site': site_names,
                                             'noise': error_vals,
                                             'samp_years': samp_years_vals,
                                             'samp_per_year': samp_per_year_vals,
                                             'initial_conc': initial_conc_vals,
                                             'reduction': red_vals,
                                             'previous_slope': previous_slope_vals,
                                             'mrt': mrt_vals,
                                             'mrt_p1': mrt_p1_vals,
                                             'frac_p1': frac_p1_vals,
                                             'f_p1': f_p1_vals,
                                             })
    assert len(outdata) == len(use_sites) * len(samp_durs) * len(samp_freqs)
    assert len(outdata.index.unique()) == len(outdata)
    assert len(outdata.site.unique()) == len(use_sites)
    assert len(out_use_conc) == len(outdata)
    return outdata, out_use_conc


def make_true_conc(site, reduction, samp_freq, samp_dur):
    true_conc = get_site_true_recept_conc(site, reduction=reduction)
    locs = np.arange(0, samp_dur + 1 / samp_freq, 1 / samp_freq)
    true_conc = pd.concat((true_conc, pd.Series(index=locs, data=np.nan)))
    true_conc = true_conc.loc[~true_conc.index.duplicated(keep='first')]
    true_conc = true_conc.interpolate(method='index')
    temp = true_conc.loc[locs]
    if (len(temp) - (samp_dur * samp_freq + 1)) == 1:
        temp = temp.iloc[:-1]
    assert len(temp) == samp_dur * samp_freq + 1
    return temp


def make_no_trend_meta_data():
    use_sites = get_final_sites()
    metadata = get_n_metadata()
    use_sites = use_sites[metadata.loc[use_sites, 'slope_yr'] <= 0]

    out_use_conc = []
    idvs = []
    site_names = []
    error_vals = []
    samp_years_vals = []
    samp_per_year_vals = []
    initial_conc_vals = []
    red_vals = []
    previous_slope_vals = []
    mrt_vals = []
    mrt_p1_vals = []
    frac_p1_vals = []
    f_p1_vals = []
    for samp_dur, samp_freq, red in itertools.product(samp_durs, samp_freqs, reductions):
        site_names.extend(use_sites)
        idvs.extend(pd.Series(use_sites) + f'_{samp_dur}_{samp_freq}')
        error_vals.extend(metadata.loc[use_sites, 'noise'])
        samp_years_vals.extend([samp_dur] * len(use_sites))
        samp_per_year_vals.extend([samp_freq] * len(use_sites))
        initial_conc_vals.extend(metadata.loc[use_sites, f'conc_{start_year}'])
        red_vals.extend([red] * len(use_sites))
        previous_slope_vals.extend(metadata.loc[use_sites, 'slope_yr'])
        mrt_vals.extend(metadata.loc[use_sites, 'age_mean'])
        mrt_p1_vals.extend(metadata.loc[use_sites, 'age_mean'])
        frac_p1_vals.extend(metadata.loc[use_sites, 'frac_1'])
        f_p1_vals.extend(metadata.loc[use_sites, 'f_p1'])

        # get true conc
        for site in use_sites:
            temp = make_true_conc(site, samp_freq=samp_freq, samp_dur=samp_dur, reduction=red)
            out_use_conc.append(temp)

    outdata = pd.DataFrame(index=idvs, data={'site': site_names,
                                             'noise': error_vals,
                                             'samp_years': samp_years_vals,
                                             'samp_per_year': samp_per_year_vals,
                                             'initial_conc': initial_conc_vals,
                                             'reduction': red_vals,
                                             'previous_slope': previous_slope_vals,
                                             'mrt': mrt_vals,
                                             'mrt_p1': mrt_p1_vals,
                                             'frac_p1': frac_p1_vals,
                                             'f_p1': f_p1_vals,
                                             })
    assert len(outdata) == len(use_sites) * len(samp_durs) * len(samp_freqs)
    assert len(outdata.index.unique()) == len(outdata)
    assert len(outdata.site.unique()) == len(use_sites)
    assert len(out_use_conc) == len(outdata)
    return outdata, out_use_conc


def get_no_trend_detection_power(recalc=False):
    save_path = generated_data_dir.joinpath('no_trend_detection_power.hdf')
    if save_path.exists() and not recalc:
        return pd.read_hdf(save_path, key='power')
    print('running: get_no_trend_detection_power')
    dpc_notrend = DetectionPowerCalculator(significance_mode='mann-kendall',
                                           nsims=1000,
                                           min_p_value=0.05,
                                           min_samples=5,
                                           expect_slope=-1,
                                           efficent_mode=True,
                                           ncores=ncores, print_freq=200)

    metadata, true_conc_ts_vals = make_no_trend_meta_data()

    data = dpc_notrend.mulitprocess_power_calcs(
        outpath=None,
        id_vals=metadata.index,
        error_vals=metadata.noise,
        mrt_model_vals='pass_true_conc',
        true_conc_ts_vals=true_conc_ts_vals,
        seed=65468,
        run=not test_dcp
    )
    if data is not None:
        temp = data.loc[:, 'python_error']
        data = data.dropna(axis=1, how='all')
        data['python_error'] = temp
        data = data.merge(metadata, left_index=True, right_index=True)
        data.to_hdf(save_path, key='power', complib='zlib', complevel=9)

    return data


def get_trend_detection_power(recalc=False):
    save_path = generated_data_dir.joinpath('trend_detection_power.hdf')
    if save_path.exists() and not recalc:
        return pd.read_hdf(save_path, key='power')
    print('running: get_trend_detection_power')
    dpc_trend = DetectionPowerCalculator(significance_mode='n-section-mann-kendall',
                                         nsims=1000,
                                         min_p_value=0.05, min_samples=5,
                                         expect_slope=(1, -1), nparts=2, min_part_size=5,
                                         no_trend_alpha=0.50, nsims_pettit=2000, efficent_mode=True,
                                         mpmk_check_step=1, mpmk_efficent_min=10, mpmk_window=0.03,
                                         ncores=ncores, print_freq=200)

    metadata, true_conc_ts_vals = make_trend_meta_data()

    data = dpc_trend.mulitprocess_power_calcs(
        outpath=None,
        id_vals=metadata.index,
        error_vals=metadata.noise,
        mrt_model_vals='pass_true_conc',
        true_conc_ts_vals=true_conc_ts_vals,
        seed=65468,
        run=not test_dcp
    )
    if data is not None:
        temp = data.loc[:, 'python_error']
        data = data.dropna(axis=1, how='all')
        data['python_error'] = temp
        data = data.merge(metadata, left_index=True, right_index=True)
        data.to_hdf(save_path, key='power', complib='zlib', complevel=9)

    return data


def get_no_trend_detection_power_no_noise(recalc=False):
    save_path = generated_data_dir.joinpath('Noise_free_no_trend_detection_power.hdf')
    if save_path.exists() and not recalc:
        return pd.read_hdf(save_path, key='power')
    print('running: get_no_trend_detection_power_no_noise')
    dpc_notrend = DetectionPowerCalculator(significance_mode='mann-kendall',
                                           nsims=1,
                                           min_p_value=0.05,
                                           min_samples=5,
                                           expect_slope=-1,
                                           efficent_mode=True,
                                           ncores=ncores, print_freq=200)
    metadata, true_conc_ts_vals = make_no_trend_meta_data()

    data = dpc_notrend.mulitprocess_power_calcs(
        outpath=None,
        id_vals=metadata.index,
        error_vals=0,
        mrt_model_vals='pass_true_conc',
        true_conc_ts_vals=true_conc_ts_vals,
        seed=65468,
        run=not test_dcp
    )
    if data is not None:
        temp = data.loc[:, 'python_error']
        data = data.dropna(axis=1, how='all')
        data['python_error'] = temp
        data = data.merge(metadata, left_index=True, right_index=True)
        data.to_hdf(save_path, key='power', complib='zlib', complevel=9)

    return data


def get_trend_detection_power_no_noise(recalc=False):
    save_path = generated_data_dir.joinpath('Noise_free_trend_detection_power.hdf')
    if save_path.exists() and not recalc:
        return pd.read_hdf(save_path, key='power')

    print('running: get_trend_detection_power_no_noise')
    dpc_trend = DetectionPowerCalculator(significance_mode='n-section-mann-kendall',
                                         nsims=1,
                                         min_p_value=0.05, min_samples=5,
                                         expect_slope=(1, -1), nparts=2, min_part_size=5,
                                         no_trend_alpha=0.50, nsims_pettit=2000, efficent_mode=True,
                                         mpmk_check_step=1, mpmk_efficent_min=10, mpmk_window=0.03,
                                         ncores=ncores, print_freq=200)
    metadata, true_conc_ts_vals = make_trend_meta_data()

    data = dpc_trend.mulitprocess_power_calcs(
        outpath=None,
        id_vals=metadata.index,
        error_vals=0,
        mrt_model_vals='pass_true_conc',
        true_conc_ts_vals=true_conc_ts_vals,
        seed=65468,
        run=not test_dcp
    )
    if data is not None:
        temp = data.loc[:, 'python_error']
        data = data.dropna(axis=1, how='all')
        data['python_error'] = temp
        data = data.merge(metadata, left_index=True, right_index=True)
        data.to_hdf(save_path, key='power', complib='zlib', complevel=9)

    return data


def make_check_all_detection_powers(recalc=False):
    funcs = [
        get_no_trend_detection_power_no_noise,
        get_trend_detection_power_no_noise,
        get_no_trend_detection_power,
        get_trend_detection_power,
    ]
    out = {}
    for func in funcs:
        print(f'running: {func.__name__}')
        t = func(recalc=recalc)
        out[func.__name__] = t

    for k, v in out.items():
        assert v.python_error.notna().sum() == 0, f'python error for {k}: {v.python_error.unique()}'


def get_all_plateau_sites():
    data = get_trend_detection_power()
    data = data.groupby('site')['power'].sum()
    return data.index[np.isclose(data, 0)]


def get_plateau_power(recalc=False):
    save_path = generated_data_dir.joinpath('plateau_detection_power.hdf')
    if save_path.exists() and not recalc:
        return pd.read_hdf(save_path, key='power')
    outdata = []
    for freq in samp_freqs:
        print(f'running: get_plateau_power {freq=}')
        dpc_trend = DetectionPowerCalculator(significance_mode='n-section-mann-kendall',
                                             nsims=1000,
                                             min_p_value=0.05, min_samples=10,
                                             expect_slope=(1, 0), nparts=2, min_part_size=5 * freq,
                                             no_trend_alpha=0.5, nsims_pettit=2000, efficent_mode=False,
                                             mpmk_check_step=1, mpmk_efficent_min=10, mpmk_window=0.03,
                                             ncores=ncores, print_freq=200)

        metadata, use_conc = make_trend_meta_data()
        use_sites = get_all_plateau_sites()
        idx = np.in1d(metadata.site, use_sites) & (metadata.samp_per_year == freq)
        metadata = metadata.loc[idx]
        use_conc = [e for i, e in zip(idx, use_conc) if i]

        data = dpc_trend.mulitprocess_power_calcs(
            outpath=None,
            id_vals=metadata.index,
            error_vals=metadata.noise,
            mrt_model_vals='pass_true_conc',
            true_conc_ts_vals=use_conc,
            seed=65468,
            run=not test_dcp
        )
        if data is not None:
            temp = data.loc[:, 'python_error']
            data = data.dropna(axis=1, how='all')
            data['python_error'] = temp
            data = data.merge(metadata, left_index=True, right_index=True)
            outdata.append(data)

    outdata = pd.concat(outdata)
    outdata.to_hdf(save_path, key='power', complib='zlib', complevel=9)

    return outdata


def run_check_plateau_power(recalc=False):
    v = get_plateau_power(recalc=recalc)
    python_errors = v.python_error.unique()
    for p in python_errors:
        print(p)
        print('\n\n\n\n')
    assert len(python_errors) == 0, f'{len(python_errors)} unique python errors, see above'
    pass


test_dcp = False
ncores = 8
if __name__ == '__main__':
    # already run: make_check_all_detection_powers(True)
    run_check_plateau_power(recalc=True)
