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


def temp_correct_power(power, power_temp_coeff, cell_temp, base_temp=25):
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
    corr_power = (
        power /
        (1 + ((power_temp_coeff / 100) * (cell_temp - base_temp)))
    )
    return corr_power


def back_of_module_temp(
    poa, temp_amb, wind_speed, module_type="glass_cell_poly", racking="open_rack"
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
    a = EMP_HEAT_COEFF[racking][module_type]["a"]
    b = EMP_HEAT_COEFF[racking][module_type]["b"]
    return poa * np.exp(a + b * wind_speed) + temp_amb


def cell_temp(bom, poa, module_type="glass_cell_poly", racking="open_rack"):
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
    return bom + (poa / 1000) * EMP_HEAT_COEFF[racking][module_type]["del_tcnd"]


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

def pvsyst_rear_irradiance(globbak, backshd):
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
