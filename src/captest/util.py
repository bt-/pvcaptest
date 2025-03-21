import warnings
import re
import json
import yaml
import numpy as np
import pandas as pd


def read_json(path):
    with open(path) as f:
        json_data = json.load(f)
    return json_data


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


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
    str or numeric
        If the `string_output` is True and the most common timestep is an integer
        in the specified units then a valid pandas frequency or offset alias is
        returned.
        If `string_output` is false, then a numeric value is returned.
    """
    units_abbrev = {
        'D': 'D',
        'M': 'M',
        'Y': 'Y',
        'h': 'H',
        'm': 'min',
        's': 'S'
    }
    common_timestep = data.index.to_series().diff().mode().values[0]
    common_timestep_tdelta = common_timestep.astype('timedelta64[m]')
    freq =common_timestep_tdelta / np.timedelta64(1, units)
    if string_output:
        try:
            return str(int(freq)) + units_abbrev[units]
        except:
            return str(freq) + units_abbrev[units]
    else:
        return freq


def reindex_datetime(data, file_name=None, report=False):
    """
    Find dataframe index frequency and reindex to add any missing intervals.

    Sorts index of passed dataframe before reindexing.

    Parameters
    ----------
    data : DataFrame
        DataFrame to be reindexed.
    file_name : str, default None   
        Name of file being reindexed. Used for warning message.

    Returns
    -------
    Reindexed DataFrame
    """
    data_index_length = data.shape[0]
    df = data.copy()
    df.sort_index(inplace=True)
    print('before calling get common timestep')
    freq_str = get_common_timestep(data, string_output=True)
    print(freq_str)
    full_ix = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq_str)
    try:
        df = df.reindex(index=full_ix)
    except ValueError:
        duplicated = df.index.duplicated()
        dropped_indices = df[duplicated].index
        # warning prints out of order in jupyter lab but not ipython, jupyter lab issue
        warnings.warn(
            f'Dropping duplicate indices from {file_name} before reindexing: {dropped_indices}',
            UserWarning,
        )
        df = df[~duplicated] # drop rows with duplicate indices before reindexing
        df = df.reindex(index=full_ix)
    df_index_length = df.shape[0]
    missing_intervals = df_index_length - data_index_length

    if report:
        print('Frequency determined to be ' + freq_str + ' minutes.')
        print('{:,} intervals added to index.'.format(missing_intervals))
        print('')

    return df, missing_intervals, freq_str

def generate_irr_distribution(
    lowest_irr,
    highest_irr,
    rng=np.random.default_rng(82)
):
    """
    Create a list of increasing values similar to POA irradiance data.

    Default parameters result in increasing values where the difference
    between each subsquent value is randomly chosen from the typical range
    of steps for a POA tracker.

    Parameters
    ----------
    lowest_irr : numeric
        Lowest value in the list of values returned.
    highest_irr : numeric
        Highest value in the list of values returned.
    rng : Numpy Random Generator
        Instance of the default Generator.

    Returns
    -------
    irr_values : list
    """
    irr_values = [lowest_irr, ]
    possible_steps = (
        rng.integers(1, high=8, size=10000)
        + rng.random(size=10000)
        - 1
    )
    below_max = True
    while below_max:
        next_val = irr_values[-1] + rng.choice(possible_steps, replace=False)
        if next_val >= highest_irr:
            below_max = False
        else:
            irr_values.append(next_val)
    return irr_values


def tags_by_regex(tag_list, regex_str):
    regex = re.compile(regex_str, re.IGNORECASE)
    return [tag for tag in tag_list if regex.search(tag) is not None]


def append_tags(sel_tags, tags, regex_str):
    new_list = sel_tags.copy()
    new_list.extend(tags_by_regex(tags, regex_str))
    return new_list


def update_by_path(dictionary, path, new_value=None, convert_callable=False):
    """
    Update a nested dictionary value by following a path list.
    
    Parameters
    ----------
    dictionary : dict
        The dictionary to update
    path : list
        A list representing the path to the target key
    new_value : optional
        The new value to set (if None and convert_callable=True,
        will convert existing tuple to function name)
    convert_callable : bool, optional
        If True and new_value is None, converts tuple to function name
    
    Returns
    -------
    updated_dictionary : dict
        The updated dictionary
    """
    # Get a reference to the current level in the dictionary
    current = dictionary
    
    # Navigate to the parent of the target key
    for key in path[:-1]:
        current = current[key]
    
    # If convert_callable is True and no new value provided, convert existing tuple
    if convert_callable and new_value is None:
        target_value = current[path[-1]]
        if isinstance(target_value, tuple) and callable(target_value[0]):
            current[path[-1]] = target_value[0].__name__
    else:
        # Update the target key with the new value
        current[path[-1]] = new_value
    
    return dictionary


def process_reg_cols(
        original_calc_params,
        calc_params=None,
        key_id=None,
        dict_path=None,
        cd=None,
    ):
    """
    Recursively process a regression columns dictionary that includes calculated parameters.

    The regression parameters dictionary attribute of CapData can be defined with a
    nested structure which includes tuples with two values where the first is a
    CapData method to calculate a new value (column of Data attribute) and the second
    is a dictionary of the kwargs to be passed to the function.

    An example tuple:
    (back_of_module_temp, {'poa': 'irr_poa', 'temp_amb':'temp_amb', 'wind_speed':'wind_speed'})
    
    Where back_of_module_temp is a CapData method that accepts the kwargs poa, temp_amb,
    and wind_speed, which have the values (column group ids) irr_poa, temp_amb, wind_speed,
    respectively.
    
    The dictionary passed to `original_calc_params` may be nested like this example:

    calc_params_map = {
        'power_tc': (CapData.temp_correct_power, { # power_tc_calc_bom
            'power': 'real_pwr_mtr',
            'cell_temp': (CapData.cell_temp, { # cell_temp_calc_bom
                'poa': 'irr_poa',
                'bom': (CapData.back_of_module_temp, { # calc_bom
                    'poa':'irr_poa', 'temp_amb':'temp_amb', 'wind_speed':'wind_speed'})
            })
        }),
    } 
    
    This function will start at the bottom of nested dictionaries and progressively
    call the functions with the kwargs replacing the function tuples with the function
    names (also the name of the new columns added to CapData.data attribute).

    NOTE: This function is expected to be called after the CapData.agg_sensors method,
    which will update the column group keys in the regression_cols attribute with the
    names of the aggregated columns.
    
    Parameters
    ----------
    original_calc_params : dict
        The original dictionary to be modified
    calc_params : dict or tuple
        The current level of the dictionary being processed
    key_id : str
        The key ID of the current level
    dict_path : list
        The path to the current level in the dictionary
    
    Returns:
        Dictionary of functions to run
    """
    if calc_params is None:
        calc_params = original_calc_params
    
    if dict_path is None:
        dict_path = []
    
    if isinstance(calc_params, dict):
        print('-' * 40)
        print('Calc params is a DICT')
        print(dict_path)
        for calc_param_id, calc_inputs in calc_params.items():
            print(calc_param_id)
            print(calc_inputs)
            print(type(calc_inputs))
            if isinstance(calc_inputs, tuple):
                print('value is tuple going in level')
                new_path = dict_path + [calc_param_id]
                process_reg_cols(
                    original_calc_params,
                    calc_inputs,
                    key_id=calc_param_id,
                    dict_path=new_path,
                    cd=cd,
                )
    elif isinstance(calc_params, tuple):
        print('=' * 40)
        print('calc params is a tuple')
        print(dict_path)
        print(calc_params)
        func = calc_params[0]
        if isinstance(calc_params[0], str) and isinstance(calc_params[1], str):
            print('Found bottom - column groups to aggregate')
            print('+' * 40)
            agg_name = cd.agg_group(group_id=calc_params[0], agg_func=calc_params[1])
            update_by_path(original_calc_params, dict_path, agg_name)
            process_reg_cols(original_calc_params, cd=cd)
        if isinstance(calc_params[1], dict):
            print('Testing dict second value of tuple for more tuples')
            if all([isinstance(values, str) for values in calc_params[1].values()]):
                print('Found bottom')
                print('+' * 40)
                # Need to add call to func here passing kwargs
                # The functions need to modify CapData.Data and add the result in a new column named func.__name__
                # The functions should be CapData methods wrapping the functions in the prtest module
                # args or kwargs that are not Series of Data should be attributes of the CapData instance
                print(f'calc_params[1]: {calc_params[1]}')
                func(cd, **calc_params[1])
                # Update the original calc_params dictionary at the current path
                update_by_path(original_calc_params, dict_path, func.__name__)
                # Recursive call to reprocess again with the modified reg_cols dict
                # Effect is to process the next layer up in the dict
                process_reg_cols(original_calc_params, cd=cd)
            else:
                print('Not bottom, go down level')
                print(calc_params[1])
                new_path = dict_path + [1]
                process_reg_cols(
                    original_calc_params,
                    calc_params[1],
                    key_id=key_id,
                    dict_path=new_path,
                    cd=cd,
                )
        elif isinstance(calc_params[1], tuple):
            new_path = dict_path + [1]
            process_reg_cols(
                original_calc_params,
                calc_params[1],
                key_id=key_id,
                dict_path=new_path,
                cd=cd,
            )
