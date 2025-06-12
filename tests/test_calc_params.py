import pytest
import pandas as pd

from captest import calcparams


class TestTempCorrectPower:
    """Test correction of power by temperature coefficient."""

    def test_output_type_numeric(self, capsys):
        """Check output type matches input type and check explanation output"""
        power_tc = calcparams.temp_correct_power(10, -0.37, 50)
        assert isinstance(power_tc, float)
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "temp_correct_power" column as '
            "(10) / (1 + ((-0.37 / 100) * (50 - 25)))"
        )

    def test_output_type_series(self, capsys):
        power_tc = calcparams.temp_correct_power(
            pd.Series([10, 12, 15], name="power_col"), -0.37, 50
        )
        assert isinstance(power_tc, pd.Series)
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "temp_correct_power" column as '
            "(power_col) / (1 + ((-0.37 / 100) * (50 - 25)))"
        )

    def test_high_temp_higher_power(self, capsys):
        power = 10
        corr_power = calcparams.temp_correct_power(power, -0.37, 50, verbose=False)
        assert corr_power > power
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_low_temp_lower_power(self):
        power = 10
        corr_power = calcparams.temp_correct_power(power, -0.37, 10)
        assert corr_power < power

    def test_math_numeric_power(self):
        power = 10
        corr_power = calcparams.temp_correct_power(power, -0.37, 50)
        assert pytest.approx(corr_power, 0.3) == 11.019

    def test_math_series_power(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        power = pd.Series([10, 20, 15], index=ix)
        corr_power = calcparams.temp_correct_power(power, -0.37, 50)
        assert pytest.approx(corr_power.values, 0.3) == [11.019, 22.038, 16.528]

    def test_no_temp_diff(self):
        assert calcparams.temp_correct_power(10, -0.37, 25) == 10

    def test_user_base_temp(self):
        corr_power = calcparams.temp_correct_power(10, -0.37, 30, base_temp=27.5)
        assert pytest.approx(corr_power, 0.3) == 10.093


class TestBackOfModuleTemp:
    """Test calculation of back of module (BOM) temperature from weather."""

    def test_float_inputs(self, capsys):
        assert calcparams.back_of_module_temp(800, 30, 3) == pytest.approx(48.1671)
        captured = capsys.readouterr()
        assert captured.out.rstrip("\n") == (
            'Calculating and adding "bom_temp" column as '
            "800 * e^(-3.56 + -0.075 * 3) + 30. "
            'Coefficients a and b assume "glass_cell_poly" modules and "open_rack" racking.'
        )

    def test_float_inputs_no_output(self, capsys):
        assert calcparams.back_of_module_temp(
            800, 30, 3, verbose=False
        ) == pytest.approx(48.1671)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_series_inputs(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix, name="poa")
        temp_amb = pd.Series([26, 27, 27.5], index=ix, name="temp_amb")
        wind = pd.Series([0.5, 1, 2.5], index=ix, name="wind")

        exp_results = pd.Series([48.0506544, 48.3709869, 46.6442104], index=ix)

        assert (
            pd.testing.assert_series_equal(
                calcparams.back_of_module_temp(poa, temp_amb, wind), exp_results
            )
            is None
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
        bom = calcparams.back_of_module_temp(
            800, 28, 1.5, module_type=module_type, racking=racking
        )
        assert bom == pytest.approx(expected)


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
        poa = pd.Series([805, 810, 812], index=ix)
        cell_temp = pd.Series([26, 27, 27.5], index=ix)

        assert calcparams.avg_typ_cell_temp(poa, cell_temp) == pytest.approx(26.8356)


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
