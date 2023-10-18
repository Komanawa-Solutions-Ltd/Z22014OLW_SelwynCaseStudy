"""
created matt_dumont 
on: 27/09/23
"""
import numpy as np
from matplotlib import pyplot as plt
from project_base import proj_root, unbacked_dir
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from kendall_stats import MannKendall
from site_selection.age_tracer_data import get_final_age_data

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

sw_name_mapper = {  # todo hopefully include stream names
    'sq30878': 'LII Stream u/s Pannetts Rd',
    'sq30916': 'Selwyn River u/s Coes Ford',
    'sq30976': 'Boggy Creek u/s Lake Road',
    'sq30977': 'Doyleston Drain at Drain Rd',
    'sq30992': 'Harts Creek d/s Lower Lake Rd',
    'sq32872': 'Halswell River at River Road bridge',
    'sq33468': 'Silverstream u/s Selwyn River confl',
    'sq34538': 'Lee River u/s Brooklands Farm bridge',
    'sq34540': 'Waikekewai Creek u/s Gullivers Road',
    'sq35586': 'Mathias Stream at gauging site',
}


def get_paired_wells():
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


def get_all_n_data(recalc=False):
    save_path = unbacked_dir.joinpath('all_n_data.hdf')
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
        out_paired = get_paired_wells()
        outdata['isite_id'] = outdata['site_id'].replace({k: i for i, k in enumerate(outdata['site_id'].unique())})
        outdata['site_id'] = outdata['site_id'].replace(out_paired)
        assert not any(np.in1d(outdata['site_id'], out_paired.keys()))

        # keynote adjust step change from matched sites offset visually
        outdata.loc[outdata['isite_id'] == 27, 'n_conc'] += -0.6
        add_manual_outlier(outdata)
        outlier_id_lof(outdata)
        outdata.to_hdf(save_path, 'ndata', complib='zlib', complevel=9)
    else:
        outdata = pd.read_hdf(save_path, 'ndata')

    return outdata


def outlier_id_lof(ndata, contaminations=None):
    if contaminations is None:
        contaminations = ['auto']
    assert isinstance(ndata, pd.DataFrame)
    refs = ndata['site_id'].unique()
    for ref in refs:
        idx = ndata['site_id'] == ref
        if idx.sum() < 5:
            continue
        X = ndata.loc[idx, ['datetime', 'n_conc']]
        X.loc[:, 'datetime'] = (X.datetime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        X = X.values

        # normalize data
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        assert np.isnan(X).sum() == 0
        for cont in contaminations:
            clf = LocalOutlierFactor(contamination=cont, n_jobs=1)
            y_pred = clf.fit_predict(X)
            ndata.loc[idx, f'unsup_outlier_{cont}'] = y_pred < 0

    return ndata


def get_n_metadata(recalc=False):
    save_path = unbacked_dir.joinpath('all_n_metadata.hdf')
    if not save_path.exists() or recalc:
        ndata = get_all_n_data(recalc=recalc)
        ngroup = ndata.groupby('site_id')
        ['min', 'max', 'mean', 'std', 'count']
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
        paired = get_paired_wells()
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

        # KEYNOTE manual ages
        manual_age_changes = {
            'm36_4655': (44.5, 'minimum age'),
            'm36_5255': (44.5, 'minimum age'),
            'm36_2679': (20, 'from M36_0160'),
            'm36_3596': (46.5, 'from m36_269'),
            'm36_3467': (20.5, 'from m36 5190'),
            'm36_3588': (20.5, 'from m36 5190'),
            'l36_0200': (20, 'gut feel'),
            'm36_3683': (20.5, 'from m36 5190'),
            'm36_0456': (12, 'FROM mean of m365190 and burnam bores'),
            'l35_0910': (60, 'from nearby deep wells (100m rather than 200m)'),
            'm36_0271': (25, 'from manual interpretation')
        }
        for k, (age, comment) in manual_age_changes.items():
            outdata.loc[k, 'age_mean'] = age
            outdata.loc[k, 'age_median'] = age
            outdata.loc[k, 'age_comment'] = comment

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
        lisa_data = pd.read_csv(proj_root.joinpath('original_data/n_metadata_lisa.csv'), index_col=0)
        lisa_data['LS_keep'] = lisa_data['LS_keep'] != 'FALSE'
        out_paired = get_paired_wells()
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
        outdata.to_hdf(save_path, 'nmetadata', complib='zlib', complevel=9)
    else:
        outdata = pd.read_hdf(save_path, 'nmetadata')

    return outdata


def get_outlier_free_ndata(recalc=False):
    ndata = get_all_n_data(recalc=recalc)


def plot_from_metadata(metadata, outdir):
    from kendall_stats import MannKendall
    outdir.mkdir(exist_ok=True, parents=True)
    ndata = get_all_n_data()
    for site in metadata.index:
        site_type = ndata.loc[ndata['site_id'] == site, 'type'].unique()[0]
        all_data = ndata[ndata['site_id'] == site].set_index('datetime')
        if len(all_data)<3:
            continue
        t = MannKendall(all_data['n_conc'])
        fig, ax, leg_data = t.plot_data()
        mdist = metadata.loc[site, 'age_dist']
        if mdist == 0:
            lag_key = 'lag_at_site'
        else:
            lag_key = f'lag_within_{mdist}m_+-_{metadata.loc[site, "age_depth"]}m_depth'

        ax.set_title(f'{site_type} {site}, depth={metadata.loc[site, "depth"]:.0f}m\n'
                     f'{lag_key}\n'
                     f'mrt={metadata.loc[site, "age_mean"]:.2f} years\n'
                     f'mk={metadata.loc[site, "mk_trend"]}, p={metadata.loc[site, "mk_p"]:.2f}')
        ax.legend()
        fig.savefig(outdir.joinpath(f'{site}.png'))
        plt.close(fig)


def add_manual_outlier(data):
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
    for k, v in lims.items():
        idx = (data.site_id == k) & (data.n_conc < v)
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
    for k, v in lims.items():
        idx = (data.site_id == k) & (data.n_conc > v)
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
    for k, v in lims.items():
        idx = (data.site_id == k) & (data.datetime < pd.to_datetime(v))
        data.loc[idx, 'exclude_for_noise'] = True

    lims = {
        'l36_0089': '2015-01-01',
    }
    for k, v in lims.items():
        idx = (data.site_id == k) & (data.datetime > pd.to_datetime(v))
        data.loc[idx, 'exclude_for_noise'] = True


def plot_wierdi():
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


def plot_outlier_managment(metadata, outdir):
    from kendall_stats import MannKendall

    outdir.mkdir(exist_ok=True, parents=True)
    ndata = get_all_n_data()
    for site in metadata.index:
        site_type = ndata.loc[ndata['site_id'] == site, 'type'].unique()[0]
        all_data = ndata[ndata['site_id'] == site].set_index('datetime')
        exclude_mk_idx = ~(all_data['exclude_for_noise'] | all_data['always_exclude'])

        t = MannKendall(all_data.loc[exclude_mk_idx, 'n_conc'])
        fig, ax = plt.subplots(figsize=(14, 8))
        fig, ax, (handles, labels) = t.plot_data(color='b', ax=ax)
        idx = all_data['exclude_for_noise']
        sc = ax.scatter(all_data.loc[idx].index, all_data.loc[idx, 'n_conc'], color='r', label='exclude_for_noise')
        handles.append(sc)
        labels.append('exclude_for_noise')
        idx = all_data['always_exclude']
        sc = ax.scatter(all_data.loc[idx].index, all_data.loc[idx, 'n_conc'], color='k', label='always_exclude')
        handles.append(sc)
        labels.append('always_exclude')

        mdist = metadata.loc[site, 'age_dist']
        if mdist == 0:
            lag_key = 'lag_at_site'
        else:
            lag_key = f'lag_within_{mdist}m_+-_{metadata.loc[site, "age_depth"]}m_depth'

        ax.set_title(f'{site_type} {site}, depth={metadata.loc[site, "depth"]:.0f}m\n'
                     f'{lag_key}\n'
                     f'mrt={metadata.loc[site, "age_mean"]:.2f} years\n'
                     f'mk={metadata.loc[site, "mk_trend"]}, p={metadata.loc[site, "mk_p"]:.2f}')
        ax.legend(handles, labels)
        fig.savefig(outdir.joinpath(f'{site}.png'))
        plt.close(fig)


# todo look at streams in context of flow if possible
# todo all sw sites at differnet ages: [5, 10, 20, 30]

if __name__ == '__main__':
    #  plot_wierdi() manually handled
    meta = get_n_metadata(True)
    for k in ['keep0',
              'lisa_keep', 'Lisa_Comment_bool', 'keep0|lisa', '~keep0&lisa']:
        print(k, meta[k].sum())
    meta.to_csv(unbacked_dir.joinpath('n_metadata.csv'))
    age_data = get_final_age_data()
    age_data.to_csv(unbacked_dir.joinpath('n_age_data.csv'))
    plot_outlier_managment(meta.loc[meta['lisa_keep']], unbacked_dir.joinpath('n_plots_use_meta'))
    plot_from_metadata(meta.loc[~meta['lisa_keep']], unbacked_dir.joinpath('n_plots_exclude_meta'))
