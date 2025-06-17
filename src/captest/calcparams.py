"""
Functions to calculate derived values from measured data.

For example, back-of-module temperature from poa, wind speed, and ambient temp with the
Sandia module temperature model.
"""

import numpy as np

EMP_HEAT_COEFF = {
    "open_rack": {
        "glass_cell_glass": {"a": -3.47, "b": -0.0594, "del_tcnd": 3},
        "glass_cell_poly": {"a": -3.56, "b": -0.0750, "del_tcnd": 3},
        "poly_tf_steel": {"a": -3.58, "b": -0.1130, "del_tcnd": 3},
    },
    "close_roof_mount": {"glass_cell_glass": {"a": -2.98, "b": -0.0471, "del_tcnd": 1}},
    "insulated_back": {"glass_cell_poly": {"a": -2.81, "b": -0.0455, "del_tcnd": 0}},
}


def get_param_ids(params):
    """Get identifier strings for parameters.

    For each parameter, returns its name attribute if it has one,
    otherwise returns its string representation.

    Parameters
    ----------
    params : dict
        Dictionary of parameter values to get identifiers for

    Returns
    -------
    dict
        Dictionary mapping parameter names to their identifier strings
    """
    return {
        name: param.name if hasattr(param, "name") else str(param)
        for name, param in params.items()
    }


def power_temp_correct(
    data, power, cell_temp, power_temp_coeff=None, base_temp=25, verbose=True
):
    """Apply temperature correction to PV power.

    Divides `power` by the temperature correction, so low power values that
    are above `base_temp` will be increased and high power values that are
    below the `base_temp` will be decreased.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    power : str
        The column name of the data attribute with the power to correct.
    cell_temp : str
        Name of the column in `data` containing the cell temperature (in Celsius) used
        to calculate temperature differential from the `base_temp`.
    power_temp_coeff : numeric
        Module power temperature coefficient as percent per degree celsius.
        Ex. -0.36
    base_temp : numeric, default 25
        Base temperature (in Celsius) to correct power to. Default is the
        STC of 25 degrees Celsius.

    Returns
    -------
    Series
        Power corrected for temperature.
    """
    if verbose:
        print(
            'Calculating and adding "temp_correct_power" column as '
            f"({power}) / (1 + (({power_temp_coeff} / 100) * "
            f"({cell_temp} - {base_temp})))"
        )
    power = data[power]
    cell_temp = data[cell_temp]
    return power / (1 + ((power_temp_coeff / 100) * (cell_temp - base_temp)))


def bom_temp(
    data,
    poa=None,
    temp_amb=None,
    wind_speed=None,
    module_type="glass_cell_poly",
    racking="open_rack",
    verbose=True,
):
    """Calculate back of module temperature from measured weather data.

    Calculate back of module temperature from POA irradiance, ambient
    temperature, wind speed (at height of 10 meters), and empirically
    derived heat transfer coefficients.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    poa : str
        Column name for POA irradiance in W/m^2.
    temp_amb : str
        Column name for Ambient temperature in degrees C.
    wind_speed : str
        Column name for Measured wind speed (m/sec) corrected to measurement height of
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
    a = EMP_HEAT_COEFF[racking][module_type]["a"]
    b = EMP_HEAT_COEFF[racking][module_type]["b"]
    if verbose:
        print(
            'Calculating and adding "bom_temp" column as '
            f"{poa} * e^({a} + {b} * {wind_speed}) + {temp_amb}. "
            f'Coefficients a and b assume "{module_type}" modules and "{racking}" racking.'
        )
    return data[poa] * np.exp(a + b * data[wind_speed]) + data[temp_amb]


def cell_temp(
    data, bom, poa, module_type="glass_cell_poly", racking="open_rack", verbose=True
):
    """Calculate cell temp from BOM temp, POA, and heat transfer coefficient.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    bom : str
        Column name for back of module temperature (degrees C). Strictly following the NREL
        procedure this value would be obtained from the `back_of_module_temp`
        function.

        Alternatively, a measured BOM temperature may be used.

        Refer to p.7 of NREL Weather Corrected Performance Ratio Report.
    poa : str
        Column name for POA irradiance in W/m^2.
    module_type : str, default 'glass_cell_poly'
        Any of glass_cell_poly, glass_cell_glass, or 'poly_tf_steel'.
    racking: str, default 'open_rack'
        Any of 'open_rack', 'close_roof_mount', or 'insulated_back'
    verbose : bool, default True
        By default prints explanation of calculation. Set to False for no output
        message.

    Returns
    -------
    Series
        Cell temperatures.
    """
    if verbose:
        print(
            'Calculating and adding "cell_temp" column using the Sandia temperature '
            f'model assuming "{module_type}" module type and "{racking}" racking '
            f'from the "{bom}" and "{poa}" columns.'
        )
    bom_data = data[bom]
    poa_data = data[poa]
    return (
        bom_data + (poa_data / 1000) * EMP_HEAT_COEFF[racking][module_type]["del_tcnd"]
    )


def avg_typ_cell_temp(data, poa, cell_temp, verbose=True):
    """Calculate irradiance weighted cell temperature.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    poa : str
        Column name for POA irradiance (W/m^2).
    cell_temp : str
        Column name for Cell temperature for each interval (degrees C).

    Returns
    -------
    float
        Average irradiance-weighted cell temperature.
    """
    return (data[poa] * data[cell_temp]).sum() / data[poa].sum()


def rpoa_pvsyst(data, globbak="GlobBak", backshd="BackShd", verbose=True):
    """Calculate the sum of PVsyst's global rear irradiance and rear shading and IAM losses.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance containing PVsyst 8760 data.
    globbak : str, default 'GlobBak'
        Column name for global rear irradiance (W/m^2).
    backshd : str, default 'BackShd'
        Column name for rear shading and IAM losses (W/m^2).
    verbose : bool, default True
        Set to False to not print calculation explanation.

    Returns
    -------
    Series
        Sum of global rear irradiance and rear shading and IAM losses.
    """
    if verbose:
        print(
            'Calculating and adding "rpoa_pvsyst" column as ' f"{globbak} + {backshd}. "
        )
    return data[globbak] + data[backshd]


def e_total(
    data, poa, rpoa, bifaciality=0.7, bifacial_frac=1, rear_shade=0, verbose=True
):
    """
    Calculate total irradiance from POA and rear irradiance.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the source data for calculations. Usually the `data` attribute
        of a CapData instance.
    poa : str
        Column name for POA irradiance (W/m^2).
    rpoa : str
        Column name for rear irradiance (W/m^2).
    bifaciality : numeric, default 0.7
        Bifaciality factor.
    bifacial_frac : numeric, default 1
        Fraction of total array nameplate power that is bifacial. Pass to calculate
        total plane of array irradiance for plants with a mix of monofacial and
        bifacial modules.
    rear_shade : numeric, default 0
        Fraction of rear irradiance that is lost due to shading. Set to decimal
        fraction, e.g. 0.12, to include in calculation of `e_total`.

    Returns
    -------
    numeric or Series
        Total plane of array irradiance.
    """
    if verbose:
        print(
            'Calculating and adding "e_total" column as '
            f'{poa} + {rpoa} * '
            f'{bifaciality} * {bifacial_frac} * '
            f'(1 - {rear_shade})'
        )
    return data[poa] + data[rpoa] * bifaciality * bifacial_frac * (1 - rear_shade)
