"""
created matt_dumont 
on: 27/09/23
"""
import numpy as np

from project_base import proj_root, unbacked_dir
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def get_all_n_data(filtered=True, recalc=False):
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

        # todo look for outliers in the dataset!, intial run doesnt' look great... probably need to do it by site if at all
        outlier_id_lof(outdata)

        # todo save as hdf once I get finished
    else:
        outdata = pd.read_hdf(save_path, 'ndata')

    if filtered:
        # todo filter sites here
        raise NotImplementedError
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
            ndata.loc[idx, f'outlier_{cont}'] = y_pred < 0

    return ndata


def get_n_metadata(filtered=True, recalc=False):
    save_path = unbacked_dir.joinpath('all_n_metadata.hdf')
    if not save_path.exists() or recalc:
        ndata = get_all_n_data(filtered=False)
        outdata = ndata.groupby('site_id').agg({'n_conc': ['min', 'max', 'mean', 'std', 'count'],
                                                'datetime':['min', 'max']})
        # todo flatten columns

        outdata['years_sampled'] = (outdata['datetime']['max'] - outdata['datetime']['min']).dt.days/365
        outdata['samples_per_year_mean'] = outdata['n_conc']['count'] / outdata['years_sampled']

        # todo filter sites here
        temp = (
                (outdata['n_conc']['count'] > 5)
                & (outdata['years_sampled'] > 5)
                & (outdata['n_conc']['max'] > 3)
                & (outdata['datetime']['min'] < pd.Timestamp('2010-01-01'))
        )

        # todo get variation, slopes, mannkendall, length, duration etc.
        # todo add ages here
        raise NotImplementedError

    else:
        outdata = pd.read_hdf(save_path, 'nmetadata')

    if filtered:
        raise NotImplementedError
    return outdata


if __name__ == '__main__':
    get_n_metadata()
