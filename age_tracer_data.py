"""
created matt_dumont 
on: 27/09/23
"""
import pandas as pd
from project_base import proj_root, unbacked_dir


# todo Lisa's age tracer data does not have the paramters... see if I can pull from the OLW data, e.g. compare
#  site lists and ages...


def get_provided_age_data():
    data = pd.read_excel(proj_root.joinpath('original_data',
                                            'Age_Tracer_PowerBI_Selwyn_Waihora_extracted25Sept2023_checked.xlsx'),
                         sheet_name='Export', comment='#')
    data['site_id'] = data['Site ID'].str.replace('/', '_').str.lower()

    return data['site_id'].dropna().unique()


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


def check_missing():
    prov_sites = get_provided_age_data()
    olw_meta, olw_data = get_olw_age_data()
    olw_sites = olw_meta.index.unique()
    missing = set(prov_sites) - set(olw_sites)
    pd.Series(list(missing)).to_csv(unbacked_dir.joinpath('missing_ages.csv')
                                    , index=False, header=False)
    pd.Series(olw_sites).to_csv(unbacked_dir.joinpath('olw_sites.csv')
                                , index=False, header=False)
    assert len(missing) == 0, f'missing {missing}'
    # todo many missing, need more info from ECAN

if __name__ == '__main__':
    check_missing()