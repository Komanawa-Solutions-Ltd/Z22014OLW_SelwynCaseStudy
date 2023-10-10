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
        assert (nztmx == 1).all()
        assert (nztmy == 1).all()
        outdata['nztmx'] = ngroup['nztmx'].mean()
        outdata['nztmy'] = ngroup['nztmy'].mean()
        outdata['type'] = ngroup['type'].first()

        depth = ngroup['depth'].nunique()
        assert (depth == 1).all()
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
            sites = outdata.index[outdata['age_mean'].isna()]
            for site in sites:
                site_x = outdata.loc[site, 'nztmx']
                site_y = outdata.loc[site, 'nztmy']
                site_depth = outdata.loc[site, 'depth']
                dists = (np.sqrt((ages['nztmx'] - site_x) ** 2 + (ages['nztmy'] - site_y) ** 2)) ** 0.5
                depths = np.abs(ages['depth'] - site_depth)
                idx = ((dists <= dist) & (depths <= depth))
                if idx.sum() == 0:
                    continue
                outdata.loc[site, 'age_dist'] = dist
                outdata.loc[site, 'age_depth'] = depth
                for c in keep_age_cols:
                    outdata.loc[site, c] = ages.loc[idx, c].mean()

        # todo filter sites here (ages), I think I'm done with this +- lisa comments
        filter_idx = (
                (outdata['ncount'] > 5)
                & (outdata['years_sampled'] > 5)
                & (outdata['n95'] > 3)
                & (outdata['nmedian'] > 2.4)
                & (outdata['datetime_min'] < pd.Timestamp('2010-01-01'))
                & (outdata['datetime_max'] > pd.Timestamp('2010-01-01'))
                & (outdata['age_mean'] < 100)
        )
        # todo save when done
    else:
        outdata = pd.read_hdf(save_path, 'nmetadata')
        filter_idx = pd.read_hdf(save_path, 'filter')

    return outdata, filter_idx


def get_outlier_free_ndata(recalc=False):
    ndata = get_all_n_data(recalc=recalc)


def plot_from_metadata(metadata, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    ndata = get_all_n_data()
    for site in metadata.index:
        fig, ax = plt.subplots(figsize=(8, 10))
        site_type = ndata.loc[ndata['site_id'] == site, 'type'].unique()[0]
        for outlier, c in zip([True, False], ['r', 'b']):
            site_data = ndata[(ndata['site_id'] == site) & (ndata['unsup_outlier_auto'] == outlier)]
            ax.scatter(site_data['datetime'], site_data['n_conc'], color=c, label='outlier' if outlier else 'inlier')

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


if __name__ == '__main__':
    meta, filt = get_n_metadata()
    print(filt.sum())
    meta['keep'] = filt
    meta['dont_keep'] = ~filt
    meta.to_csv(unbacked_dir.joinpath('n_metadata.csv'))
    plot_from_metadata(meta.loc[filt], unbacked_dir.joinpath('n_plots_use'))
    plot_from_metadata(meta.loc[~filt], unbacked_dir.joinpath('n_plots_exclude'))
