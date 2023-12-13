"""
Template created by matt_dumont
on: 22/03/22
"""
from pathlib import Path

try:
    from kslcore import KslEnv

    kslenv_loaded = True
except ModuleNotFoundError:
    kslenv_loaded = False

precision = 2
project_name = 'Z22014OLW_SelwynCaseStudy'
proj_root = Path(__file__).parent  # base of git repo
generated_data_dir = proj_root.joinpath('GeneratedData')
generated_data_dir.mkdir(exist_ok=True)
if kslenv_loaded:
    unbacked_dir = KslEnv.unbacked.joinpath(project_name)
    unbacked_dir.mkdir(exist_ok=True)
    project_dir = KslEnv.shared_drive('Z22014OLW_OLWGroundwater').joinpath('Selwyn_subproject')
else:
    unbacked_dir = Path.home().joinpath(f'{project_name}_unbacked')
    unbacked_dir.mkdir(exist_ok=True)
    project_dir = None
