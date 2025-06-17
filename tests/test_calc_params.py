import pytest
import pandas as pd

from captest import calcparams


class TestTempCorrectPower:
    """Test correction of power by temperature coefficient."""

    def test_output_type_series(self, capsys):
        df = pd.DataFrame({"power_col": [10, 12, 15], "cell_temp_col": [50, 50, 50]})
        power_tc = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert isinstance(power_tc, pd.Series)
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "temp_correct_power" column as '
            "(power_col) / (1 + ((-0.37 / 100) * (cell_temp_col - 25)))"
        )

    def test_high_temp_higher_power(self, capsys):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [50]})
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37, verbose=False
        )
        assert corr_power.iloc[0] > df["power_col"].iloc[0]
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_low_temp_lower_power(self):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [10]})
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert corr_power.iloc[0] < df["power_col"].iloc[0]

    def test_math_series_power(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        df = pd.DataFrame(
            {"power_col": [10, 20, 15], "cell_temp_col": [50, 50, 50]}, index=ix
        )
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert pytest.approx(corr_power.values, 0.3) == [11.019, 22.038, 16.528]

    def test_no_temp_diff(self):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [25]})
        corrected_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37
        )
        assert corrected_power.iloc[0] == 10

    def test_user_base_temp(self):
        df = pd.DataFrame({"power_col": [10], "cell_temp_col": [30]})
        corr_power = calcparams.power_temp_correct(
            df, "power_col", "cell_temp_col", power_temp_coeff=-0.37, base_temp=27.5
        )
        assert pytest.approx(corr_power.iloc[0], 0.3) == 10.093


class TestBomTemp:
    """Test calculation of back of module (BOM) temperature from weather."""

    def test_no_output_when_verbose_false(self, capsys):
        """Ensure bom_temp does not print when verbose is False"""
        df = pd.DataFrame({
            "poa": [800],
            "temp_amb": [25],
            "wind": [1.0],
        })
        _ = calcparams.bom_temp(df, "poa", "temp_amb", "wind", verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_dataframe_inputs(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        df = pd.DataFrame(
            {
                "poa": [805, 810, 812],
                "temp_amb": [26, 27, 27.5],
                "wind": [0.5, 1, 2.5],
            },
            index=ix,
        )

        exp_results = pd.Series([48.0506544, 48.3709869, 46.6442104], index=ix)

        pd.testing.assert_series_equal(
            calcparams.bom_temp(df, "poa", "temp_amb", "wind"), exp_results
        )
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "bom_temp" column as '
            "poa * e^(-3.56 + -0.075 * wind) + temp_amb. "
            'Coefficients a and b assume "glass_cell_poly" modules and "open_rack" racking.'
        )

    @pytest.mark.parametrize(
        "racking, module_type, expected",
        [
            ("open_rack", "glass_cell_glass", 50.77154),
            ("open_rack", "glass_cell_poly", 48.33028),
            ("open_rack", "poly_tf_steel", 46.82361),
            ("close_roof_mount", "glass_cell_glass", 65.86252),
            ("insulated_back", "glass_cell_poly", 72.98647),
        ],
    )
    def test_emp_heat_coeffs(self, racking, module_type, expected):
        # create single-row DataFrame
        df = pd.DataFrame({
            "poa": [800],
            "temp_amb": [28],
            "wind": [1.5],
        })
        bom = calcparams.bom_temp(
            df, "poa", "temp_amb", "wind", module_type=module_type, racking=racking, verbose=False
        )
        assert bom.iloc[0] == pytest.approx(expected)
    
    


class TestCellTemp:
    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix)
        temp_bom = pd.Series([26, 27, 27.5], index=ix)
        df = pd.DataFrame({"poa": poa, "bom_temp": temp_bom}, index=ix)

        exp_results = pd.Series([28.415, 29.43, 29.936], index=ix)

        pd.testing.assert_series_equal(
            calcparams.cell_temp(df, "bom_temp", "poa"), exp_results
        )

    @pytest.mark.parametrize(
        "racking, module_type, expected",
        [
            ("open_rack", "glass_cell_glass", pd.Series([28.415, 29.43, 29.936])),
            (
                "close_roof_mount",
                "glass_cell_glass",
                pd.Series([26.805, 27.81, 28.312]),
            ),
            ("insulated_back", "glass_cell_poly", pd.Series([26, 27, 27.5])),
        ],
    )
    def test_emp_heat_coeffs(self, racking, module_type, expected):
        # ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812])
        temp_bom = pd.Series([26, 27, 27.5])
        df = pd.DataFrame({"poa": poa, "bom_temp": temp_bom})
        ctemp = calcparams.cell_temp(
            df,
            "bom_temp",
            "poa",
            module_type=module_type,
            racking=racking,
            verbose=False,
        )
        pd.testing.assert_series_equal(ctemp, expected, check_names=False)

    def test_output_message_series_inputs(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix, name="poa")
        temp_bom = pd.Series([26, 27, 27.5], index=ix, name="bom_temp")
        df = pd.DataFrame({"poa": poa, "bom_temp": temp_bom}, index=ix)

        calcparams.cell_temp(df, "bom_temp", "poa")

        # Get captured stdout
        captured = capsys.readouterr()
        stdout_content = captured.out

        print(stdout_content)
        assert stdout_content.rstrip("\n") == (
            'Calculating and adding "cell_temp" column using the Sandia temperature '
            'model assuming "glass_cell_poly" module type and "open_rack" racking '
            'from the "bom_temp" and "poa" columns.'
        )


class TestAvgTypCellTemp:
    def test_math(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        df = pd.DataFrame(
            {
                "poa": [805, 810, 812],
                "cell_temp": [26, 27, 27.5],
            },
            index=ix,
        )

        assert calcparams.avg_typ_cell_temp(df, "poa", "cell_temp") == pytest.approx(26.8356)


class TestPVsystRearIrradiance:
    def test_float_inputs(self):
        assert calcparams.pvsyst_rear_irradiance(100, 10) == 110

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        globbak = pd.Series([100, 110, 120], index=ix)
        backshd = pd.Series([10, 15, 20], index=ix)

        exp_results = pd.Series([110, 125, 140], index=ix)

        pd.testing.assert_series_equal(
            calcparams.pvsyst_rear_irradiance(globbak, backshd), exp_results
        )


class TestEtotal:
    def test_numeric_inputs(self):
        assert calcparams.e_total(poa=100, rpoa=10) == 107

    def test_numeric_non_default_bifaciality(self):
        assert calcparams.e_total(poa=100, rpoa=10, bifaciality=0.5) == 105

    def test_numeric_non_default_bifi_frac(self):
        assert (
            calcparams.e_total(poa=100, rpoa=10, bifaciality=1, bifacial_frac=0.5)
            == 105
        )

    def test_numeric_non_default_bifaciality_and_bifacial_frac(self):
        assert (
            calcparams.e_total(poa=100, rpoa=20, bifaciality=0.5, bifacial_frac=0.5)
            == 105
        )

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([100, 110, 120], index=ix)
        rear = pd.Series([100, 150, 200], index=ix)

        exp_results = pd.Series([170, 215, 260], index=ix)

        pd.testing.assert_series_equal(
            calcparams.e_total(poa=poa, rpoa=rear), exp_results, check_dtype=False
        )

    def test_rear_shade(self):
        assert calcparams.e_total(poa=100, rpoa=20, rear_shade=0.5) == 107
