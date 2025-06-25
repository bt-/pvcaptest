import pytest
import copy
import numpy as np
import pandas as pd

from captest import util

ix = pd.date_range(start="1/1/2021 12:00", freq="H", periods=3)

ix_5min = pd.date_range(start="1/1/2021 12:00", freq="5min", periods=3)


class TestGetCommonTimestep:
    def test_output_type_str(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=True)
        assert isinstance(time_step, str)

    def test_output_type_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=False)
        assert isinstance(time_step, np.float64)

    def test_hours_string(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=True)
        assert time_step == "1H"

    def test_hours_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=False)
        assert time_step == 1.0

    def test_minutes_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="m", string_output=False)
        assert time_step == 60.0

    def test_mixed_intervals(self):
        df = pd.DataFrame(
            {"a": np.ones(120)},
            index=pd.date_range(start="1/1/21", freq="1min", periods=120),
        )
        assert df.index.is_monotonic_increasing
        print(df)
        df_gaps = pd.concat(
            [
                df.loc["1/1/21":"1/1/21 00:10", :],
                df.loc["1/1/21 00:15":"1/1/21 00:29", :],
                df.loc["1/1/21 00:31":, :],
            ]
        )
        assert df_gaps.shape[0] < df.shape[0]
        time_step = util.get_common_timestep(df_gaps, units="m", string_output=False)
        assert time_step == 1
        time_step = util.get_common_timestep(df_gaps, units="m", string_output=True)
        assert time_step == "1min"


@pytest.fixture
def reindex_dfs():
    df1 = pd.DataFrame(
        {"a": np.full(4, 5)},
        index=pd.date_range(start="1/1/21", end="1/1/21 00:15", freq="5min"),
    )
    df2 = pd.DataFrame(
        {"a": np.full(4, 5)},
        index=pd.date_range(start="1/1/21 00:30", end="1/1/21 00:45", freq="5min"),
    )
    return (df1, df2)


class TestReindexDatetime:
    def test_adds_missing_intervals(self, reindex_dfs):
        """Check that missing intervals in the index are added to the dataframe."""
        df1, df2 = reindex_dfs
        df = pd.concat([df1, df2])
        (df_reindexed, missing_intervals, freq_str) = util.reindex_datetime(df)
        assert df_reindexed.shape[0] == 10

        df = pd.concat([df2, df1])  # reverse order, check sorting
        (df_reindexed, missing_intervals, str) = util.reindex_datetime(df)
        assert df_reindexed.shape[0] == 10

    def test_drops_duplicate_indices(self):
        """
        Check that duplicate indices are dropped before reindexing.

        Use Nov 3, 2025 which has the 1AM hour repeated due to daylight saving time.
        """
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6]},
            index=pd.to_datetime(
                [
                    "2025-11-03 00:00",
                    "2025-11-03 01:00",
                    "2025-11-03 01:00",
                    "2025-11-03 02:00",
                    "2025-11-03 03:00",
                    "2025-11-03 04:00",
                ]
            ),
        )
        with pytest.warns(UserWarning):
            (df_reindexed, missing_intervals, freq_str) = util.reindex_datetime(df)
        assert df_reindexed.index.is_unique
        assert df_reindexed.shape[0] == 5


@pytest.fixture
def nested_calc_dict():
    """Create a nested dictionary for testing update_by_path."""

    class DummyCapData(object):
        def __init__(self):
            self.data = pd.DataFrame()

        def test_func1(self, **kwargs):
            self.test_func1_kwargs = kwargs
            self.data["test_func1"] = np.full(10, 1)

        def test_func2(self, **kwargs):
            self.test_func2_kwargs = kwargs
            self.data["test_func2"] = np.full(10, 2)

        def test_func3(self, **kwargs):
            self.test_func3_kwargs = kwargs
            self.data["test_func3"] = np.full(10, 3)

        def test_func4(self, **kwargs):
            self.test_func4_kwargs = kwargs
            self.data["test_func4"] = np.full(10, 4)

        def agg_group(self, group_id, agg_func, **kwargs):
            self.agg_group_kwargs = kwargs
            col_name = group_id + "_" + agg_func
            self.data[col_name] = np.full(10, 5)
            return col_name

        def custom_param(self, func, *args, **kwargs):
            setattr(self, f"{func.__name__}_custom_kwargs", kwargs.copy())
            func(self, **kwargs)

    dummy_cd = DummyCapData()
    dummy_cd.column_groups = {
        "real_pwr_mtr": ["metered_power_kw"],
        "irr_poa": ["pyran1", "pyran2"],
        "temp_amb": ["temp_amb1", "temp_amb2"],
        "wind_speed": ["wind_speed1", "wind_speed2"],
        "irr_rpoa": ["irr_rpoa1", "irr_rpoa2"],
    }

    test_dict = {
        "power_tc": (
            DummyCapData.test_func1,
            {
                "power": "real_pwr_mtr",
                "cell_temp": (
                    DummyCapData.test_func2,
                    {
                        "poa": ("irr_poa", "mean"),
                        "bom": (
                            DummyCapData.test_func3,
                            {
                                "poa": ("irr_poa", "mean"),
                                "temp_amb": ("temp_amb", "mean"),
                                "wind_speed": ("wind_speed", "mean"),
                            },
                        ),
                    },
                ),
            },
        ),
        "irr_total": (
            DummyCapData.test_func4,
            {
                "poa": ("irr_poa", "mean"),
                "rpoa": ("irr_rpoa", "mean"),
            },
        ),
    }
    return (dummy_cd, test_dict)


class TestUpdateByPath:
    """Test the update_by_path function."""

    def test_update_by_path_pass_new_value(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict, ["power_tc", 1, "cell_temp", 1, "bom"], new_value="temp_bom"
        )
        assert updated_dict["power_tc"][1]["cell_temp"][1]["bom"] == "temp_bom"

    def test_update_by_path_convert_callable(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict,
            ["power_tc", 1, "cell_temp", 1, "bom"],
            new_value=None,
            convert_callable=True,
        )
        assert updated_dict["power_tc"][1]["cell_temp"][1]["bom"] == "test_func3"

    def test_update_by_path_convert_callable_with_new_value(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict,
            ["power_tc", 1, "cell_temp", 1, "bom"],
            new_value="temp_bom",
            convert_callable=True,
        )
        assert updated_dict["power_tc"][1]["cell_temp"][1]["bom"] == "temp_bom"

    def test_update_by_path_convert_callable_short_path(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict, ["power_tc"], new_value=None, convert_callable=True
        )
        assert updated_dict["power_tc"] == "test_func1"


class TestProcessRegCols:
    """Test the process_reg_cols function."""

    def test_modifies_original_calc_params(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        dummy_cd.regression_cols = copy.deepcopy(test_dict)
        util.process_reg_cols(test_dict, cd=dummy_cd)
        expected_modified_reg_cols = {
            "power_tc": "test_func1",
            "irr_total": "test_func4",
        }
        # Check that methods of the DummyCapData instance are called with the
        # correct kwargs in the correct order based on columns added to the
        # data DataFrame attribute and the kwargs attributes
        assert isinstance(dummy_cd.data, pd.DataFrame)
        print(dummy_cd.data)
        print(dummy_cd.regression_cols)
        assert dummy_cd.data.shape == (10, 8)
        expected_columns = pd.Index(
            [
                "irr_poa_mean",
                "temp_amb_mean",
                "wind_speed_mean",
                "test_func3",
                "test_func2",
                "test_func1",
                "irr_rpoa_mean",
                "test_func4",
            ]
        )
        assert dummy_cd.data.columns.equals(expected_columns)
        assert dummy_cd.test_func1_kwargs == {
            "power": "metered_power_kw",
            "cell_temp": "test_func2",
            "verbose": True,
        }
        assert dummy_cd.test_func2_kwargs == {
            "poa": "irr_poa_mean",
            "bom": "test_func3",
            "verbose": True,
        }
        assert dummy_cd.test_func3_kwargs == {
            "poa": "irr_poa_mean",
            "temp_amb": "temp_amb_mean",
            "wind_speed": "wind_speed_mean",
            "verbose": True,
        }
        # check that reg_cols is rolled up all the way correctly
        for k, v in expected_modified_reg_cols.items():
            assert k in test_dict
            assert v == test_dict[k]


class TestGetCommonTimestep:
    def test_output_type_str(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=True)
        assert isinstance(time_step, str)

    def test_output_type_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=False)
        assert isinstance(time_step, np.float64)

    def test_hours_string(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=True)
        assert time_step == "1H"

    def test_hours_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=False)
        assert time_step == 1.0

    def test_minutes_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="m", string_output=False)
        assert time_step == 60.0


class TestParseRegressionFormula:
    def test_astm(self):
        lhs, rhs = util.parse_regression_formula(
            "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
        )
        assert lhs == ["power"]
        assert rhs == ["poa", "t_amb", "w_vel"]

    def test_power_temp_corr_poa(self):
        lhs, rhs = util.parse_regression_formula("power_tc ~ poa")
        assert lhs == ["power_tc"]
        assert rhs == ["poa"]

    def test_power_temp_corr_poa_intercept(self):
        lhs, rhs = util.parse_regression_formula("power_tc ~ poa - 1")
        assert lhs == ["power_tc"]
        assert rhs == ["poa"]

    def test_power_temp_corr_poa_rpoa(self):
        lhs, rhs = util.parse_regression_formula("power_tc ~ poa + rpoa")
        assert lhs == ["power_tc"]
        assert rhs == ["poa", "rpoa"]

    def test_outboard_poa_total(self):
        lhs, rhs = util.parse_regression_formula(
            "power ~ poa_total + I(poa_total * fpoa) + I(poa_total * rpoa) +"
            "I(poa_total * t_amb) + I(poa_total * w_vel)"
        )
        assert lhs == ["power"]
        assert rhs == ["poa_total", "fpoa", "rpoa", "t_amb", "w_vel"]

    def test_outboard_poa_rpoa_separate(self):
        lhs, rhs = util.parse_regression_formula(
            "power ~ (poa + rpoa) * (1 + poa + rpoa + t_amb + w_vel - 1)"
        )
        assert lhs == ["power"]
        assert rhs == ["poa", "rpoa", "t_amb", "w_vel"]
