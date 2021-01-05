import warnings

import numpy as np
import pandas as pd
import param
from scipy import stats

from captest import capdata


emp_heat_coeff = {
    'open_rack': {
        'glass_cell_glass': {
            'a': -3.47,
            'b': -0.0594,
            'del_tcnd': 3
        },
        'glass_cell_poly': {
            'a': -3.56,
            'b': -0.0750,
            'del_tcnd': 3
        },
        'poly_tf_steel': {
            'a': -3.58,
            'b': -0.1130,
            'del_tcnd': 3
        },
    },
    'close_roof_mount': {
        'glass_cell_glass': {
            'a': -2.98,
            'b': -0.0471,
            'del_tcnd': 1
        }
    },
    'insulated_back': {
        'glass_cell_poly': {
            'a': -2.81,
            'b': -0.0455,
            'del_tcnd': 0
        }
    }
}


def get_common_timestep(data, units='m', string_output=True):
    """
    Get the most commonly occuring timestep of data as frequency string.
    Parameters
    ----------
    data : Series or DataFrame
        Data with a DateTimeIndex.
    units : str, default 'm'
        String representing date/time unit, such as (D)ay, (M)onth, (Y)ear,
        (h)ours, (m)inutes, or (s)econds.
    string_output : bool, default True
        Set to False to return a numeric value.
    Returns
    -------
    str
        frequency string
    """
    units_abbrev = {
        'D': 'Day',
        'M': 'Months',
        'Y': 'Year',
        'h': 'hours',
        'm': 'minutes',
        's': 'seconds'
    }
    common_timestep = stats.mode(np.diff(data.index.values))[0][0]
    common_timestep_tdelta = common_timestep.astype('timedelta64[m]')
    freq = common_timestep_tdelta / np.timedelta64(1, units)
    if string_output:
        return str(freq) + ' ' + units_abbrev[units]
    else:
        return freq


def temp_correct_power(power, power_temp_coeff, cell_temp, base_temp=25):
    """Apply temperature correction to PV power.

    Parameters
    ----------
    power : numeric or Series
        PV power (in watts) to correct to the `base_temp`.
    power_temp_coeff : numeric
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    cell_temp : numeric or Series
        Cell temperature (in Celsius) used to calculate temperature
        differential from the `base_temp`.
    base_temp : numeric, default 25
        Base temperature (in Celsius) to correct power to. Default is the
        STC of 25 degrees Celsius.

    Returns
    -------
    type matches `power`
        Power corrected for temperature.
    """
    corr_power = power * (1 - (power_temp_coeff / 100) * (base_temp - cell_temp))
    return corr_power


def back_of_module_temp(
    poa,
    temp_amb,
    wind_speed,
    module_type='glass_cell_poly',
    racking='open_rack'
):
    """Calculate back of module temperature from measured weather data.

    Calculate back of module temperature from POA irradiance, ambient
    temperature, wind speed (at height of 10 meters), and empirically
    derived heat transfer coefficients.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    poa : numeric or Series
        POA irradiance in W/m^2.
    temp_amb : numeric or Series
        Ambient temperature in degrees C.
    wind_speed : numeric or Series
        Measured wind speed (m/sec) corrected to measurement height of
        10 meters.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'

    Returns
    -------
    numeric or Series
        Back of module temperatures.
    """
    a = emp_heat_coeff[racking][module_type]['a']
    b = emp_heat_coeff[racking][module_type]['b']
    return poa * np.exp(a + b * wind_speed) + temp_amb


def cell_temp(bom, poa, module_type='glass_cell_poly', racking='open_rack'):
    """Calculate cell temp from BOM temp, POA, and heat transfer coefficient.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    bom : numeric or Series
        Back of module temperature (degrees C). Strictly followin the NREL
        procedure this value would be obtained from the `back_of_module_temp`
        function.

        Alternatively, a measured BOM temperature may be used.

        Refer to p.7 of NREL Weather Corrected Performance Ratio Report.
    poa : numeric or Series
        POA irradiance in W/m^2.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'

    Returns
    -------
    numeric or Series
        Cell temperature(s).
    """
    return bom + (poa / 1000) * emp_heat_coeff[racking][module_type]['del_tcnd']


def avg_typ_cell_temp(poa, cell_temp):
    """Calculate irradiance weighted cell temperature.

    Parameters
    ----------
    poa : Series
        POA irradiance (W/m^2).
    cell_temp : Series
        Cell temperature for each interval (degrees C).

    Returns
    -------
    float
        Average irradiance-weighted cell temperature.
    """
    return (poa * cell_temp).sum() / poa.sum()

"""DIRECTLY BELOW DRAFT PR FUNCTION TO DO ALL VERSIONS OF CALC
DECIDED TO BREAK INTO SMALLER FUNCTIONS, LEAVE TEMPORARILY"""
def perf_ratio(
    ac_energy,
    dc_nameplate,
    poa,
    temp_corrected=False,
    bom_as_cell_temp=False,
    power_temp_coeff=None,
    cell_temp=None,
    temp_amb=None,
    wind_speed=None,
    module_type='glass_cell_poly',
    racking='open_rack',
    degradation=None,
    year=None,
    availability=None,
):
    """Calculate performance ratio.

    Parameters
    ----------
    ac_energy : Series
        Measured energy production (kWh) from system meter.
    dc_nameplate : numeric
        Summation of nameplate ratings (W) for all installed modules of system
        under test.
    poa : Series
        POA irradiance (W/m^2) for each time interval of the test.
    temp_corrected : bool, default False
        Set to true to calculate a temperature corrected performance ratio.
        Must also supply `power_temp_coeff` and may optionally specify
        `cell_temp`. Setting to True and only passing `power_temp_coeff` will
        result in the `cell_temp` and `avg_typ_cell_temp` calculated follwing
        the NREL procedure.
    bom_as_cell_temp : boolean, default False
        Set to true to use a measured back of module (BOM) temperatuere as the
        cell temperature, which is used in the temperature adjustment to the
        PR result.
    power_temp_coeff : numeric, default None
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    cell_temp : Series, default None
        Cell temperature (in Celsius) used to calculate temperature
        differential from the `avg_typ_cell_temp`.
        By default, the cell temperature is calculated from
    temp_amb : Series
        Ambient temperature (degrees C) measurements.
        Argument is ignored if `bom_as_cell_temp` is True.
    wind_speed : Series
        Measured wind speed (m/sec) corrected to measurement height of
        10 meters. Argument is ignored if `bom_as_cell_temp` is True.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
        Argument is ignored if `bom_as_cell_temp` is True.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'
        Argument is ignored if `bom_as_cell_temp` is True.
    degradation : numeric, default None
        Apply a derate for degradation to the expected power (denominator).
        Must also pass specify a value for the `year` argument.
    year : numeric
        Year of operation to use in degradation calculation.
    availability : numeric or Series, default None
        Apply an adjustment for plant availability to the expected power
        (denominator).

    Returns
    -------
    numeric or tuple
        By default returns the performance ratio as a decimal. If
        `detailed_results` is True, then returns a tuple of the perfromance
        ratio and the intermediate calculation results.
    """
    pass
    # def back_of_module_temp(
    #     poa,
    #     temp_amb,
    #     wind_speed,
    #     module_type='glass_cell_poly',
    #     racking='open_rack'
    # )
    # def cell_temp(bom, poa, module_type='glass_cell_poly', racking='open_rack')
    # def avg_typ_cell_temp(poa, cell_temp)
    # def temp_correct_power(power, power_temp_coeff, cell_temp, base_temp=25)


def perf_ratio(
    ac_energy,
    dc_nameplate,
    poa,
    unit_adj=1,
    degradation=0,
    year=1,
    availability=1,
):
    """Calculate performance ratio.

    Parameters
    ----------
    ac_energy : Series
        Measured energy production (Wh) from system meter.
    dc_nameplate : numeric
        Summation of nameplate ratings (W) for all installed modules of system
        under test.
    poa : Series
        POA irradiance (W/m^2) for each time interval of the test.
    unit_adj : numeric, default 1
        Scale factor to adjust units of `ac_energy`. For exmaple pass 1000
        to convert measured energy from kWh to Wh within PR calculation.
    degradation : numeric, default None
        Apply a derate (percent, Ex: 0.5%) for degradation to the expected
        power (denominator). Must also pass specify a value for the `year`
        argument.
        NOTE: Percent is divided by 100 to convert to decimal within function.
    year : numeric
        Year of operation to use in degradation calculation.
    availability : numeric or Series, default None
        Apply an adjustment for plant availability to the expected power
        (denominator).

    Returns
    -------
    PrResults
        Instance of class PrResults.
    """
    if not isinstance(ac_energy, pd.Series):
        warnings.warn('ac_energy must be a Pandas Series.')
        return
    if not isinstance(poa, pd.Series):
        warnings.warn('poa must be a Pandas Series.')
        return
    if not ac_energy.index.equals(poa.index):
        warnings.warn('indices of poa and ac_energy must match.')
        return
    if isinstance(availability, pd.Series):
        if not availability.index.equals(poa.index):
            warnings.warn(
                'Index of availability must match the index of '
                'the poa and ac_energy.'
            )
            return

    timestep = get_common_timestep(poa, units='h', string_output=False)
    timestep_str = get_common_timestep(poa, units='h', string_output=True)

    expected_dc = (
        availability
        * dc_nameplate
        * poa / 1000
        * (1 - degradation / 100)**year
        * timestep
    )
    pr = ac_energy.sum() * unit_adj / expected_dc.sum()

    input_cd = capdata.CapData('input_cd')
    input_cd.data = pd.concat([poa, ac_energy], axis=1)

    pr_per_timestep = ac_energy * unit_adj / expected_dc
    results_data = pd.concat([ac_energy, expected_dc, pr_per_timestep], axis=1)
    results_data.columns = ['ac_energy', 'expected_dc', 'pr_per_timestep']

    results = PrResults(
        timestep=(timestep, timestep_str),
        pr=pr,
        dc_nameplate=dc_nameplate,
        input_data=input_cd,
        results_data=results_data
    )
    return results


        # df[en_dc] = avail * DC_nameplate * (df[poa_col] / 1000) * (1 - degradation)**year * timestep
    # return ((df[ac_energy_col].sum() * unit_adj) / df[en_dc].sum()) * 100

def perf_ratio_temp_corr_nrel(
    ac_energy,
    dc_nameplate,
    poa,
    power_temp_coeff=None,
    temp_amb=None,
    wind_speed=None,
    module_type='glass_cell_poly',
    racking='open_rack',
    degradation=None,
    year=None,
    availability=None,
):
    """Calculate performance ratio.

    Parameters
    ----------
    ac_energy : Series
        Measured energy production (kWh) from system meter.
    dc_nameplate : numeric
        Summation of nameplate ratings (W) for all installed modules of system
        under test.
    poa : Series
        POA irradiance (W/m^2) for each time interval of the test.
    power_temp_coeff : numeric, default None
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    temp_amb : Series
        Ambient temperature (degrees C) measurements.
        Argument is ignored if `bom_as_cell_temp` is True.
    wind_speed : Series
        Measured wind speed (m/sec) corrected to measurement height of
        10 meters. Argument is ignored if `bom_as_cell_temp` is True.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
        Argument is ignored if `bom_as_cell_temp` is True.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'
        Argument is ignored if `bom_as_cell_temp` is True.
    degradation : numeric, default None
        NOT IMPLEMENTED
        Apply a derate for degradation to the expected power (denominator).
        Must also pass specify a value for the `year` argument.
    year : numeric
        NOT IMPLEMENTED
        Year of operation to use in degradation calculation.
    availability : numeric or Series, default None
        NOT IMPLEMENTED
        Apply an adjustment for plant availability to the expected power
        (denominator).

    Returns
    -------
    """


def perf_ratio_temp_corr_meas_bom(
    ac_energy,
    dc_nameplate,
    poa,
    temp_bom,
    power_temp_coeff=None,
    degradation=None,
    year=None,
    availability=None,
):
    """Calculate PR using measured back of module temperature as cell temp.

    Parameters
    ----------
    ac_energy : Series
        Measured energy production (kWh) from system meter.
    dc_nameplate : numeric
        Summation of nameplate ratings (W) for all installed modules of system
        under test.
    poa : Series
        POA irradiance (W/m^2) for each time interval of the test.
    temp_bom : Series
        Measured back of module (BOM) temperature (degrees C), which will be
        used as the cell temperature when calculating the temperature
        adjustment to the PR.
    power_temp_coeff : numeric, default None
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    degradation : numeric, default None
        NOT IMPLEMENTED
        Apply a derate for degradation to the expected power (denominator).
        Must also pass specify a value for the `year` argument.
    year : numeric
        NOT IMPLEMENTED
        Year of operation to use in degradation calculation.
    availability : numeric or Series, default None
        NOT IMPLEMENTED
        Apply an adjustment for plant availability to the expected power
        (denominator).

    Returns
    -------
    """


class PrResults(param.Parameterized):
    """Results from a PR calculation.
    """
    dc_nameplate = param.Number(
        bounds=(0, None),
        doc=(
            'Summation of nameplate ratings (W) for all installed modules'
            ' of system.'
        )
    )
    pr = param.Number(doc='Performance ratio result decimal fraction.')
    timestep = param.Tuple(doc='Timestep of series.')
    expected_pr = param.Number(
        bounds=(0, 1),
        doc='Expected Performance ratio result decimal fraction.'
    )
    input_data = param.ClassSelector(capdata.CapData)
    results_data = param.ClassSelector(pd.DataFrame)


    def print_pr_result(self):
        """Print summary of PR result - passing / failing and by how much
        """
        if self.pr >= self.expected_pr:
            print('The test is PASSING with a measured PR of {:.2f}, '
                  'which is {:.2f} above the expected PR of {:.2f}'.format(
                    self.pr,
                    self.pr - self.expected_pr,
                    self.expected_pr))
        else:
            print('The test is FAILING with a measured PR of {:.2f}, '
                  'which is {:.2f} below the expected PR of {:.2f}'.format(
                    self.pr,
                    self.expected_pr - self.pr,
                    self.expected_pr))
"""
************************************************************************
********** BELOW FUNCTIONS ARE NOT FULLY IMPLEMENTED / TESTED **********
************************************************************************
"""
def cell_temp_old(
    df,
    bom="Tm",
    poa_col=None,
    wspd_col=None,
    ambt_col=None,
    a=-3.56,
    b=-0.0 - 0.0750,
    del_tcnd=3,
    tcell_col="Tcell",
):
    """Calculate cell temperature using thermal model presented in NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    bom : string, default 'Tm'
        Column of back of module temperature.  Default Tm uses NREL thermal model to calculate BOM temperature.
        This option can be used to specify a column of measured BOM temperatures if desired.
    """
    df[bom] = (
        df.loc[:, poa_col] * np.exp(a + b * df.loc[:, wspd_col]) + df.loc[:, ambt_col]
    )
    df[tcell_col] = df.loc[:, bom] + df.loc[:, poa_col] / 1000 * del_tcnd
    return df


def cell_typ_avg(df, poa_col=None, cellt_col=None):
    df["poa_cellt"] = df.loc[:, poa_col] * df.loc[:, cellt_col]
    cell_typ_avg = df.loc[:, "poa_cellt"].sum() / df.loc[:, poa_col].sum()
    return (cell_typ_avg, df)


def pr_test_temp_corr(
    df,
    Tcell_typ_avg,
    DC_nameplate,
    pow_coeff,
    poa_col=None,
    cellt_col="Tcell",
    ac_energy_col=None,
    timestep=0.25,
    unit_adj=1,
    en_dc="EN_DC",
    avail=1,
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------

    pow_coeff : float
        Module power coefficient in percent. Example = -0.39
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or intc
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df = df.copy()
    df[en_dc] = (
        avail
        * DC_nameplate
        * df[poa_col]
        / 1000
        * (1 - (pow_coeff / 100) * (Tcell_typ_avg - df[cellt_col]))
        * timestep
    )
    return df[ac_energy_col].sum() * unit_adj / df[en_dc].sum() * 100


def pr_test(
    df,
    DC_nameplate,
    poa_col=None,
    ac_energy_col=None,
    degradation=0.005,
    year=1,
    timestep=0.25,
    unit_adj=1,
    en_dc="EN_DC",
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or intc
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df = df.copy()
    df[en_dc] = (
        DC_nameplate * (df[poa_col] / 1000) * (1 - degradation) ** year * timestep
    )
    return ((df[ac_energy_col].sum() * unit_adj) / df[en_dc].sum()) * 100


def pr_test_pertstep(
    df,
    Tcell_typ_avg,
    DC_nameplate,
    pow_coeff,
    poa_col=None,
    cellt_col="Tcell",
    ac_energy_col=None,
    timestep=0.25,
    unit_adj=1,
    en_dc="EN_DC",
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------

    pow_coeff : float
        Module power coefficient in percent. Example = -0.39
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or int
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df[en_dc] = (
        DC_nameplate
        * df[poa_col]
        / 1000
        * (1 - (pow_coeff / 100) * (Tcell_typ_avg - df[cellt_col]))
        * timestep
    )
    return df[ac_energy_col] * unit_adj / df[en_dc] * 100


def apply_pr_test(df):
    df = cell_temp(df, poa_col="GlobInc", wspd_col="WindVel", ambt_col="TAmb")
    cell_typavg, df = cell_typ_avg(df, poa_col="GlobInc", cellt_col="Tcell")
    pr_exp = pr_test(
        df,
        cell_typavg,
        2754000,
        -0.37,
        poa_col="GlobInc",
        ac_energy_col="E_Grid",
        timestep=1,
    )
    return pr_exp, cell_typavg


# Using the annual cell_typ_avg value
def apply_pr_test_meas_annual(df):
    df = cell_temp(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        wspd_col="Rainwise Weather Station - WIND SPEED (SENSOR 1)",
        ambt_col="Rainwise Weather Station - AMBIENT TEMPERATURE (SENSOR 1)",
    )
    cell_typavg, df = cell_typ_avg(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        cellt_col="Tcell",
    )
    # annual_cell_typavg is from tmy data above
    pr_exp = pr_test(
        df,
        annual_cell_typavg,
        2754000,
        -0.37,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        ac_energy_col="PV Meter - ACTIVE POWER",
        timestep=1,
        unit_adj=1000,
    )
    return pr_exp


## example calculating PRs by week
# weekly_prs_annual_tcell_typ_avg = pd.DataFrame(meas.groupby(pd.Grouper(freq='W')).apply(apply_pr_test_meas_annual))
## example calculating PRs be month
# monthly_prs_annual_tcell_typ_avg = pd.DataFrame(meas.groupby(pd.Grouper(freq='MS')).apply(apply_pr_test_meas_annual))


def pr_test_monthly(
    df,
    Tcell_typ_avg,
    DC_nameplate,
    pow_coeff,
    poa_col=None,
    cellt_col="Tcell",
    tcell_typ_avg_col="tcell_typ_avg_monthly",
    ac_energy_col=None,
    timestep=1,
    unit_adj=1,
    en_dc="EN_DC",
):
    """Calculate temperature adjusted performance ratio.

    Parameters
    ----------

    pow_coeff : float
        Module power coefficient in percent. Example = -0.39
    timestep : float
        Fraction of hour matching time interval of data.
        This should be 1/60 for one minute data or 0.25 for fifteen minute data.
    unit_adj : float or int
        Adjustment to ac energy to convert from kW to W or other adjustment.
    """
    df[en_dc] = (
        DC_nameplate
        * df[poa_col]
        / 1000
        * (1 - (pow_coeff / 100) * (df[tcell_typ_avg_col] - df[cellt_col]))
        * timestep
    )
    return df[ac_energy_col].sum() * unit_adj / df[en_dc].sum() * 100


# Using the annual cell_typ_avg value
def apply_pr_test_meas_monthly(df):
    df = cell_temp(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        wspd_col="Rainwise Weather Station - WIND SPEED (SENSOR 1)",
        ambt_col="Rainwise Weather Station - AMBIENT TEMPERATURE (SENSOR 1)",
    )
    cell_typavg, df = cell_typ_avg(
        df,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        cellt_col="Tcell",
    )
    # annual_cell_typavg is from tmy data above
    pr_exp = pr_test_monthly(
        df,
        annual_cell_typavg,
        2754000,
        -0.37,
        poa_col="Rainwise Weather Station - TILTED IRRADIANCE (PYR 1)",
        ac_energy_col="PV Meter - ACTIVE POWER",
        timestep=1,
        unit_adj=1000,
    )
    return pr_exp
