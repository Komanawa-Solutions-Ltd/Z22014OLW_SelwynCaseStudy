"""
Template created by matt_dumont
on: 22/03/22
"""
from pathlib import Path
from kslcore import KslEnv

project_name = 'Z22014OLW_SelwynCaseStudy'
proj_root = Path(__file__).parent  # base of git repo
project_dir = KslEnv.shared_gdrive.joinpath('Z22014OLW_OLWGroundwater','Selwyn_subproject')
unbacked_dir = KslEnv.unbacked.joinpath(project_name)
unbacked_dir.mkdir(exist_ok=True)
