"""
created matt_dumont 
on: 27/09/23
"""
import numpy as np
from matplotlib import pyplot as plt
from project_base import proj_root, unbacked_dir, generated_data_dir
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from kendall_stats import MannKendall
from site_selection.age_tracer_data import get_final_age_data

start_year = 2017
end_year = 2022

age_sample_order = (
    # (dist, depth(+-))
    (500, 5),
    (1000, 5),
    (5000, 5),
    (500, 10),
    (1000, 10),
    (5000, 10),
    (7500, 5),
    (7500, 10),
    (10000, 5),
    (10000, 10),
)
sw_ages = (5, 10, 20, 30)

sw_name_mapper = {
    'sq30878': 'LII Stream-Pannetts Rd',
    'sq30916': 'Selwyn River-Coes Ford',
    'sq30976': 'Boggy Creek - Lake Rd',
    'sq30977': 'Doyleston Drain - Drain Rd',
    'sq30992': 'Harts Creek - Lower Lake Rd',
    'sq32872': 'Halswell River - River Rd',
    'sq33468': 'Silverstream - Selwyn River',
    'sq34538': 'Lee River - Brooklands Farm',
    'sq34540': 'Waikekewai Creek - Gullivers Rd',
    'sq35586': 'Mathias Stream',
}


def _get_paired_wells():
    data = pd.read_csv(proj_root.joinpath('original_data/n_metadata_lisa.csv'), index_col=0)
    paired_wells = data.loc[data['Paired well'].notna(), 'Paired well']
    paired_wells = paired_wells.rename(index={'l35_0107': 'l36_0107'})
    out_paired = {}
    for site, psite in paired_wells.to_dict().items():
        if site in out_paired.keys():
            raise ValueError
        if psite in out_paired.keys():
            if out_paired[psite] != site:
                raise ValueError
            continue
        if site == 'l37_0555':
            out_paired[psite] = 'l37_0555'
            continue

        out_paired[site] = psite
    t = set(out_paired.keys()).intersection(out_paired.values())
    assert len(t) == 0, t
    a = set(np.concatenate((paired_wells.index, paired_wells.values)))
    b = set(out_paired.keys()).union(out_paired.values())
    assert a == b, a.symmetric_difference(b)
    return out_paired


def _outlier_id_lof(ndata, contaminations=None):
    if contaminations is None:
        contaminations = ['auto']
    assert isinstance(ndata, pd.DataFrame)
    refs = ndata['site_id'].unique()
    for ref in refs:
        idx = ndata['site_id'] == ref
        if np.sum(idx) < 5:
            continue
        x = ndata.loc[idx, ['datetime', 'n_conc']]
        x.loc[:, 'datetime'] = (x.datetime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        x = x.values

        # normalize data
        x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

        assert np.isnan(x).sum() == 0
        for cont in contaminations:
            clf = LocalOutlierFactor(contamination=cont, n_jobs=1)
            y_pred = clf.fit_predict(x)
            ndata.loc[idx, f'unsup_outlier_{cont}'] = y_pred < 0

    return ndata


def _generate_noise_data(metadata):
    ndata = get_all_n_data()
    ndata = ndata.loc[~(ndata['always_exclude'] | ndata['exclude_for_noise'])]
    for site in metadata.index:
        site_ndata = ndata.loc[ndata['site_id'] == site]
        trend = metadata.loc[site, 'mk_trend']

        if trend <= 0:
            metadata.loc[site, f'conc_{start_year}'] = site_ndata['n_conc'].median()
            metadata.loc[site, 'noise'] = metadata.loc[site, 'nstd']
            metadata.loc[site, 'slope_yr'] = 0
            metadata.loc[site, 'intercept'] = np.nan
        elif pd.isna(trend):
            pass
        else:
            site_ndata['yr'] = (site_ndata['datetime'] - site_ndata['datetime'].min()).dt.days / 365.25  # convert to yr
            site_ndata = site_ndata.set_index('yr')
            mk = MannKendall(site_ndata['n_conc'])
            senslope, senintercept, lo_slope, up_slope = mk.calc_senslope()
            site_ndata['pred'] = senslope * site_ndata.index + senintercept
            site_ndata['resid'] = site_ndata['n_conc'] - site_ndata['pred']
            metadata.loc[site, 'slope_yr'] = senslope
            metadata.loc[site, 'intercept'] = senintercept
            metadata.loc[site, 'noise'] = site_ndata['resid'].std()

            xtime = (pd.to_datetime(f'{start_year}-01-01') - site_ndata['datetime'].min()).days / 365.25
            metadata.loc[site, f'conc_{start_year}'] = senslope * xtime + senintercept


def _manage_manual_age(metadata):
    # KEYNOTE manual ages
    manual_age_changes = {
        'm36_4655': (44.5, 'minimum surrounding age'),
        'm36_5255': (44.5, 'minimum surrounding age'),
        'm36_2679': (20, 'from M36_0160'),
        'm36_3596': (46.5, 'from m36_269'),
        'm36_3467': (20.5, 'from m36_5190'),
        'm36_3588': (20.5, 'from m36_5190'),
        'l36_0200': (20, 'from manual interpretation'),
        'm36_3683': (20.5, 'from m36_5190'),
        'm36_0456': (12, 'from mean of m36_5190 and Burnam bores'),
        'l35_0910': (60, 'from nearby deep wells (100m rather than 200m)'),
        'm36_0271': (25, 'from manual interpretation')
    }
    for k0, (age, comment) in manual_age_changes.items():
        metadata.loc[k0, 'age_mean'] = age
        metadata.loc[k0, 'age_median'] = age
        metadata.loc[k0, 'age_comment'] = comment


def _add_manual_outlier(data):
    data['exclude_for_noise'] = False
    data['always_exclude'] = False

    # keep greater than
    lims = {
        'l35_0171': 1.75,
        'm36_5248': 4,
        'm36_7734': 3,
        'm36_8187': 4,
        'l36_0584': 6,
        "m36_3683": 1,
    }
    for k0, v in lims.items():
        idx = (data.site_id == k0) & (data.n_conc < v)
        data.loc[idx, 'always_exclude'] = True

    # keep less than
    lims = {
        'l36_0059': 3.5,
        'l36_0682': 12,
        'l36_0725': 7,
        'l36_0871': 9.5,
        'm36_8187': 11.5,
        'm36_0456': 11,
        'l35_0190': 11,
        'l35_0205': 9,
        'l35_0596': 12,
    }
    for k0, v in lims.items():
        idx = (data.site_id == k0) & (data.n_conc > v)
        data.loc[idx, 'always_exclude'] = True

    # time management
    lims = {
        'l36_0121': '2008-02-01',
        'l35_0009': '2014-01-01',
        'l36_0477': '2000-01-01',
        'm36_0698': '1996-01-01',
        'm36_5248': '2008-01-01',
        "m36_3683": '1997-01-01',
        'l36_2122': '2006-01-01',
        'm36_8187': '2006-01-01',
    }
    for k0, v in lims.items():
        idx = (data.site_id == k0) & (data.datetime < pd.to_datetime(v))
        data.loc[idx, 'exclude_for_noise'] = True

    lims = {

    }
    for k0, v in lims.items():
        idx = (data.site_id == k0) & (data.datetime > pd.to_datetime(v))
        data.loc[idx, 'exclude_for_noise'] = True

    site = 'l36_0089'
    idx = (data.site_id == site) & (data.datetime > pd.to_datetime('2015-01-01')) & (data.n_conc < 9.15)
    data.loc[idx, 'always_exclude'] = True


def _plot_wierdi():  # manually handled
    data = get_all_n_data(True)
    sites = [
        'm36_3588',
        'm36_3683',
        'm36_4227',
        'l36_2122',
    ]
    for site in sites:
        tdata = data[data['site_id'] == site]
        fig, ax = plt.subplots()
        ax.scatter(tdata['datetime'], tdata['n_conc'], c=tdata['isite_id'])
        ax.set_title(site)
    plt.show()


def plot_single_site(site, ndata, metadata, ax=None, alpha=1, reduction=None):
    """
    plot a single site
    :param site: site id
    :param ndata: all n data (from get_all_n_data)
    :param metadata: all metadata (from get_n_metadata)
    :return: fig, ax, handles, labels
    """
    site_type = ndata.loc[ndata['site_id'] == site, 'type'].unique()[0]
    all_data = ndata[ndata['site_id'] == site].set_index('datetime')
    exclude_mk_idx = ~(all_data['exclude_for_noise'] | all_data['always_exclude'])

    t = MannKendall(all_data.loc[exclude_mk_idx, 'n_conc'])
    if ax is not None:
        assert isinstance(ax, plt.Axes)
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
    fig, ax, (handles, labels) = t.plot_data(color='b', ax=ax, alpha=alpha)
    idx = all_data['exclude_for_noise']
    sc = ax.scatter(all_data.loc[idx].index, all_data.loc[idx, 'n_conc'], color='r', label='exclude_for_noise', alpha=alpha)
    handles.append(sc)
    labels.append('exclude_for_noise')
    idx = all_data['always_exclude']
    sc = ax.scatter(all_data.loc[idx].index, all_data.loc[idx, 'n_conc'], color='k', label='always_exclude', alpha=alpha)
    handles.append(sc)
    labels.append('always_exclude')

    mdist = metadata.loc[site, 'age_dist']
    if mdist == 0:
        lag_key = 'MRT sampled'
    elif pd.notna(metadata.loc[site, 'age_comment']):
        lag_key = f'MRT inferred: {metadata.loc[site, "age_comment"]}'
    else:
        lag_key = f'MRT inferred: median within {mdist}m +- {metadata.loc[site, "age_depth"]}m depth'
    title = [

    ]
    if reduction is None:
        title.append(f'{site_type.capitalize()} {site}\n')
    else:
        title.append(f'{site_type.capitalize()} {site}, reduction={int(reduction * 100)}%\n')
    title.append(

        f'depth={metadata.loc[site, "depth"]:.0f}m, '
        f'trend={MannKendall.map_trend(metadata.loc[site, "mk_trend"])}, p={metadata.loc[site, "mk_p"]:.2f}\n'
        f'lag={metadata.loc[site, "age_mean"]:.2f} yr {lag_key}\n'
        f'noise={metadata.loc[site, "noise"]:.2f} mg/L, '
        f'slope={metadata.loc[site, "slope_yr"]:.2f} mg/L/yr, '
        f'start concentration={metadata.loc[site, f"conc_{start_year}"]:.2f} mg/L')
    ax.set_title(''.join(title))
    ax.legend(handles, labels)
    return fig, ax, handles, labels


def plot_outlier_managment(metadata, outdir):
    """
    plot the outlier data plots for all sites in metadata
    :param metadata:
    :param outdir: dir to save plots
    :return:
    """
    outdir.mkdir(exist_ok=True, parents=True)
    ndata = get_all_n_data()
    for site in metadata.index:
        fig, ax, h, l = plot_single_site(site, ndata, metadata)
        fig.tight_layout()
        fig.savefig(outdir.joinpath(f'{site}.png'))
        plt.close(fig)


def get_all_n_data(recalc=False, duplicate_strs=True):
    """
    get all n data, including stream data cleaned and ready to use with outlier flags
    :param recalc: recalc from raw data
    :param duplicate_strs: bool if true duplicate stream data for each age (so sites match)
    :return:
    """
    save_path = generated_data_dir.joinpath('all_n_data.hdf')
    if not save_path.exists() or recalc:
        raw_data = pd.read_excel(
            proj_root.joinpath(
                'original_data/GroundwaterSOE_Nitrate_timeseries_Selwyn_Waihora_exported22Sept2023.xlsx'),
            'Export')

        raw_data['type'] = 'well'
        raw_data['site_id'] = raw_data['Well ID'].str.replace('/', '_').str.lower()
        raw_data['datetime'] = pd.to_datetime(raw_data['DateTime'])
        raw_data['nztmx'] = pd.to_numeric(raw_data['NZTMX'])
        raw_data['nztmy'] = pd.to_numeric(raw_data['NZTMY'])
        raw_data['depth'] = pd.to_numeric(raw_data['Well Depth'])
        raw_data['temp_n'] = raw_data['Nitrate Nitrogen (mg/L)'].astype(str)

        str_data = pd.read_excel(
            proj_root.joinpath('original_data/20230926_SWZ_KSL.xlsx'), sheet_name='Data', comment='#')
        str_data['type'] = 'stream'
        str_data['site_id'] = str_data['Site Name'].str.replace('/', '_').str.lower()
        str_data['datetime'] = pd.to_datetime(str_data['Date'].dt.date.astype(str) + ' ' + str_data['Time'].astype(str))
        str_data['nztmx'] = pd.to_numeric(str_data['Easting'])
        str_data['nztmy'] = pd.to_numeric(str_data['Northing'])
        str_data['depth'] = 0
        str_data['temp_n'] = str_data['Nitrate-N Nitrite-N'].astype(str)

        raw_data = pd.concat([raw_data, str_data], axis=0, ignore_index=True)
        raw_data.sort_values(['site_id', 'datetime'], inplace=True)
        idx = raw_data['temp_n'].str.contains('<')
        raw_data.loc[idx, 'temp_n'] = raw_data.loc[idx, 'temp_n'].str.replace('<', '')
        raw_data['n_conc'] = pd.to_numeric(raw_data['temp_n'], errors='coerce')
        raw_data.loc[idx, 'n_conc'] = raw_data.loc[idx, 'n_conc'] / 2
        bad_idx = raw_data['n_conc'].isna() & raw_data['Nitrate Nitrogen (mg/L)'].notna()
        if bad_idx.any():
            print('bad data')
            print(raw_data.loc[bad_idx, 'Nitrate Nitrogen (mg/L)'].unique())
            raise ValueError('bad data')

        outdata = raw_data[['site_id', 'type', 'datetime', 'nztmx', 'nztmy', 'depth', 'n_conc']].dropna()

        # pair wells (from lisa scott)
        out_paired = _get_paired_wells()
        outdata['isite_id'] = outdata['site_id'].replace({k0: i for i, k0 in enumerate(outdata['site_id'].unique())})
        outdata['site_id'] = outdata['site_id'].replace(out_paired)
        assert not any(np.in1d(outdata['site_id'], list(out_paired.keys())))

        # keynote adjust step change from matched sites offset visually
        outdata.loc[outdata['isite_id'] == 27, 'n_conc'] += -0.6
        _add_manual_outlier(outdata)
        _outlier_id_lof(outdata)

        # flip to sw names
        for k0, v in sw_name_mapper.items():
            idx = outdata['site_id'] == k0
            assert idx.sum() > 0
            outdata.loc[idx, 'site_id'] = v
        outdata.to_hdf(save_path, 'ndata', complib='zlib', complevel=9)
    else:
        outdata = pd.read_hdf(save_path, 'ndata')
    if duplicate_strs:
        all_outdata = [outdata]
        for site in outdata.loc[outdata['type'] == 'stream', 'site_id'].unique():
            for age in sw_ages:
                temp = outdata.loc[outdata['site_id'] == site].copy()
                temp['site_id'] = temp['site_id'] + f' mrt-{age}'
                all_outdata.append(temp)
        outdata = pd.concat(all_outdata)

    return outdata


def get_n_metadata(recalc=False):
    """
    get metadata for all n data, cleaned with key data (e.g. noise), and keep boolean indexes
    :param recalc: bool if true recalc from raw data
    :return:
    """
    save_path = generated_data_dir.joinpath('all_n_metadata.hdf')
    if not save_path.exists() or recalc:
        ndata = get_all_n_data(recalc=recalc, duplicate_strs=False)
        ngroup = ndata.groupby('site_id')
        outdata = pd.DataFrame()
        outdata['ncount'] = ngroup['n_conc'].count()
        outdata['nmean'] = ngroup['n_conc'].mean()
        outdata['nstd'] = ngroup['n_conc'].std()
        outdata['nmin'] = ngroup['n_conc'].min()
        outdata['nmax'] = ngroup['n_conc'].max()
        outdata['n05'] = ngroup['n_conc'].quantile(0.05)
        outdata['n25'] = ngroup['n_conc'].quantile(0.25)
        outdata['nmedian'] = ngroup['n_conc'].median()
        outdata['n75'] = ngroup['n_conc'].quantile(0.75)
        outdata['n95'] = ngroup['n_conc'].quantile(0.95)
        outdata['nrange'] = outdata['nmax'] - outdata['nmin']
        outdata['datetime_min'] = ngroup['datetime'].min()
        outdata['datetime_max'] = ngroup['datetime'].max()
        outdata['years_sampled'] = (outdata['datetime_max'] - outdata['datetime_min']).dt.days / 365
        outdata['samples_per_year_mean'] = outdata['ncount'] / outdata['years_sampled']
        nztmx = ngroup['nztmx'].nunique()
        nztmy = ngroup['nztmy'].nunique()
        paired = _get_paired_wells()
        assert ((nztmx == 1) | (nztmx.index.isin(paired.values()))).all()
        assert ((nztmy == 1) | (nztmy.index.isin(paired.values()))).all()
        outdata['nztmx'] = ngroup['nztmx'].mean()
        outdata['nztmy'] = ngroup['nztmy'].mean()
        outdata['type'] = ngroup['type'].first()

        depth = ngroup['depth'].nunique()
        assert ((depth == 1) | (depth.index.isin(paired.values()))).all()
        outdata['depth'] = ngroup['depth'].mean()
        outdata.loc[outdata.type == 'stream', 'depth'] = 0

        for site in outdata.index:
            temp_n = ndata.loc[ndata['site_id'] == site, 'n_conc']
            if len(temp_n) < 5:
                continue
            mk = MannKendall(temp_n)
            outdata.loc[site, 'mk_trend'] = mk.trend
            outdata.loc[site, 'mk_p'] = mk.p

        # add ages
        ages = get_final_age_data()
        keep_age_cols = ['f_p1', 'f_p2', 'frac_1', 'mrt1', 'age_min', 'age_max', 'age_mean', 'age_std', 'age_median']
        outdata = outdata.merge(ages[keep_age_cols], how='left', left_index=True, right_index=True)
        idx = outdata['age_mean'].notna()
        outdata.loc[idx, 'age_dist'] = 0
        outdata.loc[idx, 'age_depth'] = 0

        for dist, depth in age_sample_order:
            sites = outdata.index[outdata['age_median'].isna()]
            for site in sites:
                site_x = outdata.loc[site, 'nztmx']
                site_y = outdata.loc[site, 'nztmy']
                site_depth = outdata.loc[site, 'depth']
                dists = (np.sqrt((ages['nztmx'] - site_x) ** 2 + (ages['nztmy'] - site_y) ** 2))
                depths = np.abs(ages['depth'] - site_depth)
                idx = ((dists <= dist) & (depths <= depth))
                if idx.sum() == 0:
                    continue
                outdata.loc[site, 'age_dist'] = dist
                outdata.loc[site, 'age_depth'] = depth
                for c in keep_age_cols:
                    outdata.loc[site, c] = ages.loc[idx, c].median()

        _manage_manual_age(outdata)

        sw_sites = outdata['type'] == 'stream'

        outdata.loc[sw_sites, 'age_mean'] = np.nan

        filter_idx = (
                (outdata['ncount'] > 5)
                & (outdata['years_sampled'] > 5)
                & (outdata['n95'] > 3)
                & (outdata['nmedian'] > 2.4)
                & (outdata['datetime_min'] < pd.Timestamp('2010-01-01'))
                & (outdata['datetime_max'] > pd.Timestamp('2010-01-01'))
                & ((outdata['age_mean'] < 100) | (outdata['age_mean'].isna()))
        )
        outdata['keep0'] = filter_idx

        # add in lisa data
        lisa_data = pd.read_csv(proj_root.joinpath('original_data/n_metadata_lisa.csv'))
        lisa_data['site_id'] = lisa_data['site_id'].replace(sw_name_mapper)
        lisa_data.set_index('site_id', inplace=True)
        lisa_data['LS_keep'] = lisa_data['LS_keep'] != 'FALSE'
        out_paired = _get_paired_wells()
        idx = lisa_data.index[lisa_data.index.isin(outdata.index)]
        outdata.loc[idx, 'lisa_keep'] = lisa_data.loc[idx, 'LS_keep']
        outdata['Lisa_Comment'] = ''
        outdata.loc[idx, 'Lisa_Comment'] = lisa_data.loc[idx, 'COMMENT']

        missing_idx = lisa_data.index[~lisa_data.index.isin(outdata.index)]
        missing_idx_rep = [out_paired[x] for x in missing_idx]
        outdata.loc[missing_idx_rep, 'lisa_keep'] = outdata.loc[missing_idx_rep, 'lisa_keep'] | lisa_data.loc[
            missing_idx, 'LS_keep']
        outdata.loc[missing_idx_rep, 'Lisa_Comment'] = (
                pd.Series(outdata.loc[missing_idx_rep, 'Lisa_Comment'].values)
                + pd.Series(lisa_data.loc[missing_idx, 'COMMENT'].values))

        outdata['Lisa_Comment_bool'] = outdata['Lisa_Comment'] != ''
        outdata['keep0|lisa'] = outdata['keep0'] | outdata['lisa_keep']
        outdata['~keep0&lisa'] = ~outdata['keep0'] & outdata['lisa_keep']
        assert not any(outdata.index.isin(out_paired.keys()))
        assert all(np.in1d(list(out_paired.values()), outdata.index))

        outdata['lisa_keep'] = outdata['lisa_keep'].astype(bool)
        outdata['final_keep'] = outdata['lisa_keep']

        # all sw sites at differnet ages: [5, 10, 20, 30]
        idx = outdata['type'] == 'stream'
        sw_data_org = outdata.loc[idx]
        outdata = [outdata.loc[~idx]]
        for age in sw_ages:
            sw_data = sw_data_org.copy()
            sw_data.index = sw_data.index + f' mrt-{age}'
            sw_data.loc[:, 'age_mean'] = age
            sw_data.loc[:, 'age_median'] = age
            sw_data.loc[:, 'age_comment'] = f'sw age {age}'
            outdata.append(sw_data)

        outdata = pd.concat(outdata)

        # add noise
        _generate_noise_data(outdata)

        outdata.to_hdf(save_path, 'nmetadata', complib='zlib', complevel=9)
    else:
        outdata = pd.read_hdf(save_path, 'nmetadata')
    assert isinstance(outdata, pd.DataFrame)

    # fill frac_1, f_p1
    outdata['frac_1'] = outdata['frac_1'].fillna(1)
    outdata['f_p1'] = outdata['f_p1'].fillna(outdata['f_p1'].median())

    temp = outdata.loc[outdata['final_keep'], [
        'age_mean', 'frac_1', 'f_p1', 'slope_yr', 'noise', 'nmin', 'nmax', f'conc_{start_year}', 'mk_trend'
    ]].notna().all()
    assert temp.all(), f'na data in necessary fields of final_keep: {temp}'
    return outdata


def get_final_sites():
    """
    convenience function to get final site_ids
    :return:
    """
    meta = get_n_metadata(False)
    return meta.loc[meta['final_keep']].index


if __name__ == '__main__':
    meta = get_n_metadata(True)
    for k in ['keep0',
              'lisa_keep', 'Lisa_Comment_bool', 'keep0|lisa', '~keep0&lisa']:
        print(k, meta[k].sum())
    meta.to_csv(unbacked_dir.joinpath('n_metadata.csv'))
    plot_outlier_managment(meta.loc[meta['final_keep']], unbacked_dir.joinpath('final_n_plots'))
