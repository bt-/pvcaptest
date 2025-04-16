"""
Functions to calculate derived values from measured data.

For example, back-of-module temperature from poa, wind speed, and ambient temp with the
Sandia module temperature model.
"""
import numpy as np
import pandas as pd

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


def temp_correct_power(**kwargs):
    """Apply temperature correction to PV power.

    Divides `power` by the temperature correction, so low power values that
    are above `base_temp` will be increased and high power values that are
    below the `base_temp` will be decreased.

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
    kwargs.setdefault('base_temp', 25)
    corr_power = (
        kwargs['power'] /
        (1 + ((kwargs['power_temp_coeff'] / 100) * (kwargs['cell_temp'] - kwargs['base_temp'])))
    )
    if kwargs.get('verbose', True):
        param_ids = get_param_ids(kwargs)
        print(
            'Calculating and adding "temp_correct_power" column as '
            f'({param_ids["power"]}) / (1 + (({param_ids["power_temp_coeff"]} / 100) * '
            f'({param_ids["cell_temp"]} - {param_ids["base_temp"]})))'
        )
    return corr_power


def back_of_module_temp(**kwargs):
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
    kwargs.setdefault('racking', 'open_rack')
    kwargs.setdefault('module_type', 'glass_cell_poly')
    a = EMP_HEAT_COEFF[kwargs['racking']][kwargs['module_type']]["a"]
    b = EMP_HEAT_COEFF[kwargs['racking']][kwargs['module_type']]["b"]
    if kwargs.get('verbose', True):
        param_ids = get_param_ids(kwargs)
        print(
            'Calculating and adding "bom_temp" column as '
            f'{param_ids["poa"]} * e^({a} + {b} * {param_ids["wind_speed"]}) + {param_ids["temp_amb"]}. '
            f'Coefficients a and b assume "{kwargs['module_type']}" modules and "{kwargs['racking']}" racking.'
        )
    return kwargs['poa'] * np.exp(a + b * kwargs['wind_speed']) + kwargs['temp_amb']


def cell_temp(bom, poa, module_type="glass_cell_poly", racking="open_rack", verbose=True):
    """Calculate cell temp from BOM temp, POA, and heat transfer coefficient.

    Equation from NREL Weather Corrected Performance Ratio Report.

    Parameters
    ----------
    bom : numeric or Series
        Back of module temperature (degrees C). Strictly following the NREL
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
    verbose : bool, default True
        By default prints explanation of calculation. Set to False for no output
        message.

    Returns
    -------
    numeric or Series
        Cell temperature(s).
    """
    if verbose:
        if isinstance(bom, pd.Series) and isinstance(poa, pd.Series):
            print(
                'Calculating and adding "cell_temp" column using the Sandia temperature '
                f'model assuming "{module_type}" module type and "{racking}" racking '
                f'from the "{bom.name}" and "{poa.name}" columns.'
            )
        elif isinstance(bom, pd.Series):
            print(
                'Calculating and adding "cell_temp" column using the Sandia temperature '
                f'model assuming "{module_type}" module type and "{racking}" racking '
                f'from the "{bom.name}" column and poa value provided.'
            )
        elif isinstance(poa, pd.Series):
            print(
                'Calculating and adding "cell_temp" column using the Sandia temperature '
                f'model assuming "{module_type}" module type and "{racking}" racking '
                f'from the bom value provided and "{poa.name}" column.'
            )
        else:
            print(
                'Calculating and adding "cell_temp" column using the Sandia temperature '
                f'model assuming "{module_type}" module type and "{racking}" racking '
                'from the bom and poa values provided.'
            )
    return bom + (poa / 1000) * EMP_HEAT_COEFF[racking][module_type]["del_tcnd"]


def avg_typ_cell_temp(poa, cell_temp, verbose=True):
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

def pvsyst_rear_irradiance(globbak, backshd, verbose=True):
    """Calculate the sum of PVsyst's global rear irradiance and rear shading and IAM losses.

    Parameters
    ----------
    globbak : numeric or Series
        Global rear irradiance (W/m^2).
    backshd : numeric or Series
        Rear shading and IAM losses (W/m^2).

    Returns
    -------
    numeric or Series
        Sum of global rear irradiance and rear shading and IAM losses.
    """
    return globbak + backshd

def e_total(poa, rpoa, bifaciality=0.7, bifacial_frac=1, verbose=True):
    """
    Calculate total irradiance from POA and rear irradiance.
    
    Parameters
    ----------
    poa : numeric or Series
        POA irradiance (W/m^2).
    rpoa : numeric or Series
        Rear irradiance (W/m^2).
    bifaciality : numeric, default 0.7
        Bifaciality factor.
    bifacial_frac : numeric, default 1
        Fraction of total array nameplate power that is bifacial. Pass to calculate
        total plane of array irradiance for plants with a mix of monofacial and
        bifacial modules.

    Returns
    -------
    numeric or Series
        Total plane of array irradiance.
    """
    return poa + rpoa * bifaciality * bifacial_frac
