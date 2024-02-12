"""
created matt_dumont 
on: 13/02/24
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gw_age_tools import predict_future_conc_bepm, binary_exp_piston_flow, check_age_inputs
from project_base import generated_data_dir


def mrt_explain_fig():
    once_and_future_source_conc = pd.Series({
        -50.: 1,
        -40: 1.5,
        -30: 2,
        -20: 3,
        -10: 5,
        -7: 7,
        -5: 6.5,
        -3: 6.8,
        -2: 7,
        -1: 6.2,
        0: 5.5,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=False)
    ax2.sharex(ax1)
    mrt = 10
    mrt_p1 = 10
    frac_p1 = 1
    precision = 2
    mrt_p2 = None
    f_p1 = 0.7
    f_p2 = 0.3
    mrt, mrt_p2 = check_age_inputs(mrt, mrt_p1, mrt_p2, frac_p1, precision, f_p1, f_p2)
    predict_start = -10
    predict_stop = 0
    recptor = predict_future_conc_bepm(once_and_future_source_conc, predict_start, predict_stop,
                                       mrt_p1, frac_p1, f_p1, f_p2, mrt=mrt, mrt_p2=mrt_p2, fill_value=1)
    ax1.plot(convert_to_dt(once_and_future_source_conc.index), once_and_future_source_conc.values)
    ax1.set_ylabel('Concentration')
    ax1.set_title('Source Concentration over time')

    ax3.plot(convert_to_dt(recptor.index), recptor.values, 'o-')
    ax3.set_ylabel('Concentration')
    ax3.set_title('=\n\nReceptor Concentration over time')
    ax3.scatter(convert_to_dt(-5.2), recptor.loc[-5.2], color='r', marker='o', s=200, zorder=5, label=convert_to_dt(-5.2).date())
    ax3.legend()

    ages = np.arange(0, 50 - 5.2, 0.01)
    pdf = binary_exp_piston_flow(ages, mrt_p1, mrt_p2, frac_p1, f_p1, f_p2, )
    ax2.plot(convert_to_dt(ages * -1 + -5.2), pdf, label='PDF')
    ax2.set_ylabel('Proportion of water\nfrom source at a time')
    ax2.set_title(f'X\n\nAge Distribution at {convert_to_dt(-5.2).date()}')

    fig.supxlabel('Date')
    fig.tight_layout()
    fig.savefig(generated_data_dir.joinpath('mrt_explain_fig.png'))
    plt.show()

def convert_to_dt(ages):
    return pd.to_timedelta(ages * 365, unit='D') + pd.to_datetime('2020-01-01')


if __name__ == '__main__':
    mrt_explain_fig()
