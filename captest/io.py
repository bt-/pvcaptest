# this file is formatted with black
import dateutil
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from captest.capdata import CapData
from captest.capdata import csky
from captest import columngroups as cg
from captest import util


def flatten_multi_index(columns):
    return ["_".join(col_name) for col_name in columns.to_list()]


def load_pvsyst(
    path,
    name="pvsyst",
    egrid_unit_adj_factor=None,
    set_regression_columns=True,
    **kwargs,
):
    """
    Load data from a PVsyst energy production model.

    Parameters
    ----------
    path : str
        Path to file to import.
    name : str, default pvsyst
        Name to assign to returned CapData object.
    egrid_unit_adj_factor : numeric, default None
        E_Grid will be divided by the value passed.
    set_regression_columns : bool, default True
        By default sets power to E_Grid, poa to GlobInc, t_amb to T Amb, and w_vel to
        WindVel. Set to False to not set regression columns on load.
    **kwargs
        Use to pass additional kwargs to pandas read_csv.

    Returns
    -------
    CapData

    Notes
    -----
    Standardizes the ambient temperature column name to T_Amb. v6.63 of PVsyst
    used "T Amb", v.6.87 uses "T_Amb", and v7.2 uses "T_Amb". Will change 'T Amb'
    or 'TAmb' to 'T_Amb' if found in the column names.

    """
    dirName = Path(path)

    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    for encoding in encodings:
        try:
            # there is a pandas bug prior to pandas v1.3.0 that causes the blank
            # line between the headers and data to be skipped
            # after v.1.3.0, the blank line will be loaded
            # loading headers and data separately and then combining them to avoid
            # issues with pandas versions before and after the fix
            pvraw_headers = pd.read_csv(
                dirName, skiprows=10, encoding=encoding, header=[0, 1], **kwargs
            ).columns
            pvraw_data = pd.read_csv(
                dirName, skiprows=12, encoding=encoding, header=None, **kwargs
            ).dropna(axis=0, how="all")
            pvraw = pvraw_data.copy()
            pvraw.columns = pvraw_headers
        except UnicodeDecodeError:
            continue
        else:
            break

    pvraw.columns = pvraw.columns.droplevel(1)
    dates = pvraw.loc[:, "date"]
    try:
        dt_index = pd.to_datetime(dates, format="%m/%d/%y %H:%M")
    except ValueError:
        warnings.warn(
            'Dates are not in month/day/year format. '
            'Trying day/month/year format.'
        )
        dt_index = pd.to_datetime(dates, format="%d/%m/%y %H:%M")
    pvraw.index = dt_index
    pvraw.drop("date", axis=1, inplace=True)
    pvraw = pvraw.rename(columns={"T Amb": "T_Amb"}).rename(columns={"TAmb": "T_Amb"})


    cd = CapData(name)
    pvraw.index.name = "Timestamp"
    cd.data = pvraw.copy()
    cd.data['index'] = cd.data.index.to_series().apply(
        lambda x: x.strftime('%m/%d/%Y %H %M')
    )
    if egrid_unit_adj_factor is not None:
        cd.data["E_Grid"] = cd.data["E_Grid"] / egrid_unit_adj_factor
    cd.data_filtered = cd.data.copy()
    cd.column_groups = cg.group_columns(cd.data)
    cd.trans_keys = list(cd.column_groups.keys())
    if set_regression_columns:
        cd.set_regression_cols(
            power="E_Grid", poa="GlobInc", t_amb="T_Amb", w_vel="WindVel"
        )
    return cd




def file_reader(path, **kwargs):
    """
    Read measured solar data from a csv file.

    Utilizes pandas read_csv to import measure solar data from a csv file.
    Attempts a few diferent encodings, trys to determine the header end
    by looking for a date in the first column, and concantenates column
    headings to a single string.

    Parameters
    ----------
    path : Path
        Path to file to import.
    **kwargs
        Use to pass additional kwargs to pandas read_csv.

    Returns
    -------
    pandas DataFrame
    """
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    for encoding in encodings:
        try:
            data_file = pd.read_csv(
                path,
                encoding=encoding,
                index_col=0,
                parse_dates=True,
                skip_blank_lines=True,
                low_memory=False,
                **kwargs,
            )
        except UnicodeDecodeError:
            continue
        else:
            break
    data_file.dropna(how="all", axis=0, inplace=True)
    if not isinstance(data_file.index[0], pd.Timestamp):
        for i, _indice in enumerate(data_file.index):
            try:
                isinstance(
                    dateutil.parser.parse(str(data_file.index[i])), datetime.date
                )
                header_end = i + 1
                break
            except ValueError:
                continue
        header = list(np.arange(header_end))
        data_file = pd.read_csv(
            path,
            encoding=encoding,
            header=header,
            index_col=0,
            parse_dates=True,
            skip_blank_lines=True,
            low_memory=False,
            **kwargs,
        )

    data_file = data_file.apply(pd.to_numeric)
    if isinstance(data_file.columns, pd.MultiIndex):
        data_file.columns = flatten_multi_index(data_file.columns)
    data_file = data_file.rename(columns=(lambda x: x.strip()))
    return data_file


@dataclass
class DataLoader:
    """
    Class to load SCADA data and return a CapData object.
    """

    path: str = "./data/"
    loc: Optional[dict] = field(default=None)
    sys: Optional[dict] = field(default=None)
    file_reader: object = file_reader
    files_to_load: Optional[list] = field(default=None)

    def __setattr__(self, key, value):
        if key == "path":
            value = Path(value)
        super().__setattr__(key, value)

    def set_files_to_load(self, extension="csv"):
        """
        Set `files_to_load` attribute to a list of filepaths.
        """
        self.files_to_load = [file for file in self.path.glob("*." + extension)]
        if len(self.files_to_load) == 0:
            return warnings.warn(
                "No files with .{} extension were found in the directory: {}".format(
                    extension,
                    self.path,
                )
            )

    def _reindex_loaded_files(self):
        """Reindex files to ensure no missing indices and find frequency for each file.

        Returns
        -------
        reindexed_dfs : dict
            Filenames mapped to reindexed DataFrames.
        common_freq : str
            The index frequency most common across the reindexed DataFrames.
        file_frequencies : list
            The index frequencies for each file.
        """
        reindexed_dfs = {}
        file_frequencies = []
        for name, file in self.loaded_files.items():
            current_file, missing_intervals, freq_str = util.reindex_datetime(
                file,
                report=False,
                add_index_col=True,
            )
            reindexed_dfs[name] = current_file
            file_frequencies.append(freq_str)

        unique_freq = np.unique(
            np.array([freq for freq in file_frequencies]),
            return_counts=True,
        )
        common_freq = unique_freq[0][np.argmax(unique_freq[1])]

        return reindexed_dfs, common_freq, file_frequencies

    def _join_files(self):
        """Combine the DataFrames of `loaded_files` into a single DataFrame.

        Checks if the columns of each DataFrame in `loaded_files` matches. If they do
        all match, then they will be combined along vertically along the index.

        If they do not match, then they will be combined by creating a datetime index
        that begins with the earliest datetime in all the indices to the latest datetime
        in all the indices using the most common frequency across all the indices. The
        columns will be a set of all the columns.

        Returns
        -------
        data : DataFrame
            The combined data.
        """
        all_columns = [df.columns for df in self.loaded_files.values()]
        columns_match = all(
            [pair[0].equals(pair[1]) for pair in combinations(all_columns, 2)]
        )
        all_indices = [df.index for df in self.loaded_files.values()]
        indices_match = all(
            [pair[0].equals(pair[1]) for pair in combinations(all_indices, 2)]
        )
        if columns_match and not indices_match:
            data = pd.concat(self.loaded_files.values(), axis="index")
        elif columns_match and indices_match:
            warnings.warn("Some columns contain overlapping indices.")
            data = pd.concat(self.loaded_files.values(), axis="index")
        else:
            joined_columns = pd.Index(
                set([item for cols in all_columns for item in cols])
            ).sort_values()
            data = pd.DataFrame(
                index=pd.date_range(
                    start=min([df.index.min() for df in self.loaded_files.values()]),
                    end=max([df.index.max() for df in self.loaded_files.values()]),
                    freq=self.common_freq,
                ),
                columns=joined_columns,
            )
            for file in self.loaded_files.values():
                data.loc[file.index, file.columns] = file.values
        data = data.apply(pd.to_numeric, errors="coerce")
        return data

    def load(self, extension="csv", **kwargs):
        """
        Load file(s) of timeseries data from SCADA / DAS systems.

        This is a convience function to generate an instance of DataLoader
        and call the `load` method.

        A single file or multiple files can be loaded. Multiple files will be joined together
        and may include files with different column headings. When multiple files with
        matching column headings are loaded, the individual files will be reindexed and
        then joined. Missing time intervals within the individual files will be filled,
        but missing time intervals between the individual files will not be filled.

        Parameters
        ----------
        extension : str, default "csv"
            Change the extension to allow loading different filetypes. Must also set
            the `file_reader` attribute to a function that will read that type of file.
        """
        if self.path.is_file():
            data = self.file_reader(self.path)
        elif self.path.is_dir():
            if self.files_to_load is not None:
                self.loaded_files = {
                    file.stem: self.file_reader(file) for file in self.files_to_load
                }
            else:
                self.set_files_to_load(extension=extension)
                self.loaded_files = {
                    file.stem: self.file_reader(file, **kwargs) for file in self.files_to_load
                }
            (
                self.loaded_files,
                self.common_freq,
                self.file_frequencies,
            ) = self._reindex_loaded_files()
            data = self._join_files()
            data.index.name = "Timestamp"
        self.data = data

    def sort_data(self):
        self.data.sort_index(inplace=True)

    def drop_duplicate_rows(self):
        self.data.drop_duplicates(inplace=True)

    def reindex(self):
        self.data, self.missing_intervals, self.freq_str = util.reindex_datetime(
            self.data,
            report=False,
        )


def load_data(
    path,
    group_columns=cg.group_columns,
    file_reader=file_reader,
    name="meas",
    sort=True,
    drop_duplicates=True,
    reindex=True,
    site=None,
    **kwargs,
):
    """
    Load file(s) of timeseries data from SCADA / DAS systems.

    This is a convience function to generate an instance of DataLoader
    and call the `load` method.

    A single file or multiple files can be loaded. Multiple files will be joined together
    and may include files with different column headings.

    Parameters
    ----------
    path : str
        Path to either a single file to load or a directory of files to load.
    group_columns : function or str, default columngroups.group_columns
        Function to use to group the columns of the loaded data. Function should accept
        a DataFrame and return a dictionary with keys that are ids and valeus that are
        lists of column names. Will be set to the `group_columns` attribute of the
        CapData.DataLoader object.
        Provide a string to load column grouping from a json file.
    file_reader : function, default io.file_reader
        Function to use to load an individual file. By default will use the built in
        `file_reader` function to try to load csv files. If passing a function to read
        other filetypes, the kwargs should include the filetype extension e.g. 'parquet'.
    name : str
        Identifier that will be assigned to the returned CapData instance.
    sort : bool, default True
        By default sorts the data by the datetime index from old to new.
    drop_duplicates : bool, default True
        By default drops rows of the joined data where all the columns are duplicats
        of another row. Keeps the first instance of the duplicated values. This is
        helpful if individual datafiles have overlaping rows with the same data.
    reindex : bool, default True
        By default will create a new index for the data using the earliest datetime,
        latest datetime, and the most frequent time interval ensuring there are no
        missing intervals.
    site : dict, default None
        Pass a dictionary containing site data, which will be used to generate
        modeled clear sky ghi and poa values. The clear sky irradiance values are
        added to the data and the column_groups attribute is updated to include these
        two irradiance columns. The site data dictionary should be
        {sys: {system data}, loc: {location data}}. See the capdata.csky documentation
        for the format of the system data and location data.
    **kwargs
        Passed to `DataLoader.load` Options include: sort, drop_duplicates, reindex,
        extension. See `DataLoader` for complete documentation.
    """
    dl = DataLoader(
        path=path,
        file_reader=file_reader,
    )
    dl.load(**kwargs)

    if sort:
        dl.sort_data()
    if drop_duplicates:
        dl.drop_duplicate_rows()
    if reindex:
        dl.reindex()

    cd = CapData(name)
    cd.data = dl.data.copy()
    cd.data_filtered = cd.data.copy()
    cd.data_loader = dl
    # group columns
    if callable(group_columns):
        cd.column_groups = group_columns(cd.data)
    elif isinstance(group_columns, str):
        p = Path(group_columns)
        if p.suffix == ".json":
            cd.column_groups = cg.ColumnGroups(util.read_json(group_columns))
        elif (p.suffix == ".yml") or (p.suffix == ".yaml"):
            cd.column_groups = cg.ColumnGroups(util.read_yaml(group_columns))
    if site is not None:
        cd.data = csky(cd.data, loc=site['loc'], sys=site['sys'])
        cd.data_filtered = cd.data.copy()
        cd.column_groups['irr-poa-clear_sky'] = ['poa_mod_csky']
        cd.column_groups['irr-ghi-clear_sky'] = ['ghi_mod_csky']
    cd.trans_keys = list(cd.column_groups.keys())
    cd.set_plot_attributes()
    return cd


#     loc=None,
#     sys=None,
#     """
#     Import data from csv files.
#
#     The intent of the default behavior is to combine csv files that have
#     the same columns and rows of data from different times. For example,
#     combining daily files of 5 minute measurements from the same sensors
#     for each day.
#
#     Use the path and fname arguments to specify a single file to import.
#
#     Parameters
#     ----------
#     loc : dict
#         See the csky function for details on dictionary options.
#     sys : dict
#         See the csky function for details on dictionary options.
#   """
#         if clear_sky:
#             if loc is None:
#                 warnings.warn(
#                     "Must provide loc and sys dictionary\
#                               when clear_sky is True.  Loc dict missing."
#                 )
#             if sys is None:
#                 warnings.warn(
#                     "Must provide loc and sys dictionary\
#                               when clear_sky is True.  Sys dict missing."
#                 )
#             cd.data = csky(cd.data, loc=loc, sys=sys, concat=True, output="both")
#
#     if callable(group_columns):
#         cd.column_groups = group_columns(cd.data)
#     elif isinstance(group_columns, str):
#         p = Path(group_columns)
#         if p.suffix == ".json":
#             cd.column_groups = cg.ColumnGroups(util.read_json(group_columns))
#         # elif p.suffix == '.xlsx':
#         #     cd.column_groups = "read excel file"
#
#     cd.data_filtered = cd.data.copy()
#     cd.trans_keys = list(cd.column_groups.keys())
#     return cd