"""
created matt_dumont 
on: 27/09/23
"""
import pandas as pd
from project_base import proj_root, unbacked_dir
def get_provided_age_data():
    data = pd.read_excel(proj_root.joinpath('original_data',
                                            'Age_Tracer_PowerBI_Selwyn_Waihora_extracted25Sept2023_checked.xlsx'),
                         sheet_name='Export', comment='#')
    data = data.loc[:235]
    data['site_id'] = data['Site ID'].str.replace('/', '_').str.lower()
    data.set_index('site_id', inplace=True)
    data = data.rename(columns={
        'Well Depth (m)': 'depth',
        'NZTMX': 'nztmx',
        'NZTMY': 'nztmy',
    })

    age_cols = [
        'Tritium age',
        'SF6 age',
        'CFC-12 age',
        'CFC-11 age',
        'CFC age',
        '14C age',
    ]
    keep_cols = ['depth', 'nztmx', 'nztmy'] + age_cols
    outdata = []
    for site, row in data.iterrows():
        keep = pd.DataFrame(columns=['site_id', 'age', 'depth', 'nztmx', 'nztmy', 'tracer'])
        for i, age_col in enumerate(age_cols):
            age = row[age_col]
            if pd.notna(age):
                if age in ['C', '*']:
                    continue
                keep.loc[i, 'site_id'] = site
                keep.loc[i, 'age'] = row[age_col]
                keep.loc[i, 'depth'] = row['depth']
                keep.loc[i, 'nztmx'] = row['nztmx']
                keep.loc[i, 'nztmy'] = row['nztmy']
                keep.loc[i, 'tracer'] = age_col.replace(' age', '')
        outdata.append(keep)
    outdata = pd.concat(outdata, axis=0, ignore_index=True)
    outdata['age'] = outdata['age'].astype(str)
    outdata['age'] = outdata['age'].str.replace('<', '')
    outdata['age'] = outdata['age'].str.replace('>', '')
    idx = outdata['age'].str.contains('to')
    outdata.loc[idx, 'age'] = [str(pd.to_numeric(x.split('to')).mean()) for x in outdata.loc[idx, 'age']]
    idx = outdata['age'].str.contains('or')
    outdata.loc[idx, 'age'] = [pd.to_numeric(x.split('or')).mean() for x in outdata.loc[idx, 'age']]
    t = pd.to_numeric(outdata['age'], errors='coerce')
    idx = t.isna() & outdata['age'].notna()
    if idx.any():
        for v in outdata.loc[idx, 'age'].unique():
            print(v)
    outdata['age'] = pd.to_numeric(outdata['age'], errors='raise')
    outdata['tracer'] = outdata['tracer'].str.lower()
    outdata['nztmx'] = pd.to_numeric(outdata['nztmx'])
    outdata['nztmy'] = pd.to_numeric(outdata['nztmy'])
    outdata['depth'] = pd.to_numeric(outdata['depth'])
    grouped = outdata.groupby('site_id')
    nu = grouped[['nztmx', 'nztmy', 'depth']].nunique()
    assert (nu == 1).all().all()
    out_metadata = grouped[['nztmx', 'nztmy', 'depth']].first()
    out_metadata['age_mean'] = grouped['age'].mean()
    out_metadata['age_std'] = grouped['age'].std()
    out_metadata['age_min'] = grouped['age'].min()
    out_metadata['age_max'] = grouped['age'].max()
    out_metadata['n_age'] = grouped['age'].count()
    out_metadata['tracer'] = grouped['tracer'].apply(lambda x: ','.join(x))

    return out_metadata, outdata


def get_olw_age_data():
    metadata_save_path = proj_root.joinpath('original_data/olw_data/final_age_tracer_metadata.csv')
    age_data_save_path = proj_root.joinpath('original_data/olw_data/final_age_tracer_data.csv')

    save_paths_exists = [metadata_save_path.exists(), age_data_save_path.exists()]

    metadata_types = {'ref': 'str', 'nztm_x': 'float', 'nztm_y': 'float', 'top_screen': 'float',
                      'aquifer_type': 'str', 'formation': 'str', 'alt_name': 'str', 'database': 'str',
                      'bore_depth': 'float',
                      'bottom_screen': 'float', 'site_id': 'str', 'site_location': 'str', 'screen_range': 'str',
                      'lithology': 'str', 'altitude': 'str', 'regional_council': 'str', 'report_source': 'str',
                      'region': 'str',
                      'water_level_bgl_m': 'str',
                      'f_p1': 'float',
                      'f_p2': 'float',
                      'frac_1': 'float',
                      'mrt1': 'float',
                      }

    age_data_types = {'site_name': 'str', 'unclean_age': 'str', 'standard_deviation': 'float',
                      'age': 'float', 'data_flag': 'str', 'age_type': 'str', 'alt_name': 'str', 'site_details': 'str',
                      'site_id': 'str',
                      'f_p1': 'float',
                      'f_p2': 'float',
                      'frac_1': 'float',
                      'mrt1': 'float',
                      }

    if not all(save_paths_exists):
        raise ValueError('missing save paths, from Z22014OLW_OLWGroundwater/clean_lag_data/get_cleaned_lag_data.py')
    final_metadata = pd.read_csv(metadata_save_path)
    # manage datatypes
    for k, t in metadata_types.items():
        final_metadata.loc[:, k] = final_metadata.loc[:, k].astype(t)

    final_age_data = pd.read_csv(age_data_save_path)
    # manage datatypes
    final_age_data.loc[:, 'date'] = pd.to_datetime(final_age_data.loc[:, 'date'])
    for z, y in age_data_types.items():
        final_age_data.loc[:, z] = final_age_data.loc[:, z].astype(y)

    final_metadata.set_index('ref', inplace=True)
    idx = final_metadata.bore_depth.isna() & final_metadata.top_screen.notna()
    final_metadata.loc[idx, 'bore_depth'] = final_metadata.loc[idx, 'top_screen']

    return final_metadata, final_age_data


def get_final_age_data():
    olw = get_olw_age_data()[0]
    selwyn_center = (1529387, 5172015)
    radius = 70000
    idx = ((olw.nztm_x - selwyn_center[0]) ** 2 + (olw.nztm_y - selwyn_center[1]) ** 2)**0.5 < radius
    olw = olw.loc[idx]
    provided_meta, prov_ages = get_provided_age_data()
    new_pov = set(provided_meta.index) - set(olw.index)
    new_olw = set(olw.index) - set(provided_meta.index)
    assert len(new_pov) == 0, f'new_pov {new_pov}'
    olw = olw.rename(columns={'nztm_x': 'nztmx', 'nztm_y': 'nztmy', 'bore_depth': 'depth'})
    return olw


if __name__ == '__main__':
    get_final_age_data()
