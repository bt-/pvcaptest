import pytest
import copy
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from captest import capdata as pvc
from captest import util
from captest import columngroups as cg

@pytest.fixture
def meas():
    """Create an instance of CapData with example data loaded."""
    meas = pvc.CapData('meas')
    meas.data = pd.read_csv(
        './tests/data/example_measured_data.csv',
        index_col=0,
        parse_dates=True,
        )
    meas.data_filtered = meas.data.copy(deep=True)
    meas.column_groups = cg.ColumnGroups(util.read_json(
        './tests/data/example_measured_data_column_groups.json'
    ))
    meas.trans_keys = copy.deepcopy(meas.column_groups.keys())
    meas.set_regression_cols(
        power='meter_power', poa='irr_poa_pyran', t_amb='temp_amb', w_vel='wind'
    )
    return meas

@pytest.fixture
def location_and_system():
    """Create a dictionary with a nested dictionary for location and system."""
    loc_sys = {
        'location': {
            'latitude': 30.274583,
            'longitude': -97.740352,
            'altitude': 500,
            'tz': 'America/Chicago',
        },
        'system': {
            'surface_tilt': 20,
            'surface_azimuth': 180,
            'albedo': 0.2,
        }
    }
    return loc_sys

@pytest.fixture
def nrel():
    nrel = pvc.CapData('nrel')
    nrel.data = pd.read_csv(
        './tests/data/nrel_data.csv', index_col=0, parse_dates=True
    )
    nrel.data_filtered = nrel.data.copy()
    nrel.column_groups = {
        'irr-ghi-': ['Global CMP22 (vent/cor) [W/m^2]', ],
        'irr-poa-': ['POA 40-South CMP11 [W/m^2]', ],
        'temp--': ['Deck Dry Bulb Temp [deg C]', ],
        'wind--': ['Avg Wind Speed @ 19ft [m/s]', ],
    }
    nrel.trans_keys = list(nrel.column_groups.keys())
    nrel.regression_cols = {
        'power': '', 'poa': 'irr-poa-', 't_amb': 'temp--', 'w_vel': 'wind--'
    }
    return nrel

@pytest.fixture
def pvsyst():
    # load pvsyst csv file
    df = pd.read_csv(
        './tests/data/pvsyst_example_HourlyRes_2.CSV',
        skiprows=9,
        encoding='latin1',
    ).iloc[1:, :]
    df['Timestamp'] = pd.to_datetime(df['date'])
    df = df.set_index('Timestamp', drop=True)
    df = df.drop(columns=['date']).astype(np.float64)
    df.rename(columns={'T Amb': 'T_Amb'}, inplace=True)
    # set pvsyst DataFrame to CapData data attribute
    pvsyst = pvc.CapData('pvsyst')
    pvsyst.data = df
    pvsyst.data_filtered = pvsyst.data.copy()
    pvsyst.column_groups = {
        'irr-poa-': ['GlobInc'],
        'shade--': ['FShdBm'],
        'index--': ['index'],
        'wind--': ['WindVel'],
        '-inv-': ['EOutInv'],
        'pvsyt_losses--': ['IL Pmax', 'IL Pmin', 'IL Vmax', 'IL Vmin'],
        'temp-amb-': ['T_Amb'],
        'irr-ghi-': ['GlobHor'],
        'temp-mod-': ['TArray'],
        'real_pwr--': ['E_Grid'],
    }
    pvsyst.regression_cols = {
        'power': 'real_pwr--', 'poa': 'irr-poa-', 't_amb': 'temp-amb-', 'w_vel': 'wind--'
    }
    pvsyst.trans_keys = list(pvsyst.column_groups.keys())
    return pvsyst

@pytest.fixture
def pvsyst_irr_filter(pvsyst):
    pvsyst.filter_irr(200, 800)
    pvsyst.tolerance = '+/- 5'
    return pvsyst

@pytest.fixture
def nrel_clear_sky(nrel):
    """ Modeled clear sky data was created using the pvlib fixed tilt clear sky
    models with the following parameters:
         loc = {
            'latitude': 39.742,
            'longitude': -105.18,
            'altitude': 1828.8,
            'tz': 'Etc/GMT+7'
        }
        sys = {'surface_tilt': 40, 'surface_azimuth': 180, 'albedo': 0.2}
    """
    clear_sky = pd.read_csv(
        './tests/data/nrel_data_modelled_csky.csv', index_col=0, parse_dates=True
    )
    nrel.data = pd.concat([nrel.data, clear_sky], axis=1)
    nrel.data_filtered = nrel.data.copy()
    nrel.column_groups['irr-poa-clear_sky'] = ['poa_mod_csky']
    nrel.column_groups['irr-ghi-clear_sky'] = ['ghi_mod_csky']
    nrel.trans_keys = list(nrel.column_groups.keys())
    return nrel

@pytest.fixture
def capdata_reg_result_one_coeff():
    """
    Create a CapData instance with regression results.
    """
    np.random.seed(9876789)

    meas = pvc.CapData('meas')
    meas.rc = {'x': [6]}

    nsample = 100
    e = np.random.normal(size=nsample)

    x = np.linspace(0, 10, 100)
    das_y = x * 2
    das_y = das_y + e

    das_df = pd.DataFrame({'y': das_y, 'x': x})

    das_model = smf.ols(formula='y ~ x - 1', data=das_df)

    meas.regression_results = das_model.fit()
    meas.data_filtered = pd.DataFrame()
    return meas

@pytest.fixture
def capdata_spatial():
    """
    Create a CapData object with dummy data in the data attribute.

    For each of the test regression terms (poa and t_amb), the dummy data contains three
    columns of data (e.g. poa1, poa2, poa3). The dataframe should have 3 rows of data
    with a datetime index beginning on 2023-04-01 12:00 and 1 minute frequency.
    """
    # create a datetime index
    index = pd.date_range(
        start='2023-04-01 12:00', periods=3, freq='min', tz='America/Chicago'
    )
    # create a dataframe with dummy data
    data = pd.DataFrame(
        {
            'poa1': [800, 805, 796.1],
            'poa2': [804, 781.5, 799],
            'poa3': [802.1, 799, 800],
            't_amb1': [28, 29, 30.1],
            't_amb2': [29.2, 29.4, 29],
            't_amb3': [27, 31, 30.5],
            'power': [500, 510, 505],
        },
        index=index,
    )
    # create a CapData object
    cd = pvc.CapData('test')
    # set the data attribute to the dummy data
    cd.data = data
    cd.data_filtered = cd.data.copy()
    cd.column_groups = {
        'irr_poa': ['poa1', 'poa2', 'poa3'],
        'temp_amb': ['t_amb1', 't_amb2', 't_amb3'],
    }
    cd.trans_keys = list(cd.column_groups.keys())
    cd.regression_cols = {'power': 'power', 'poa': 'irr_poa', 't_amb': 'temp_amb'}
    return cd
