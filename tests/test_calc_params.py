import pytest
import pandas as pd

from captest import calcparams

class TestTempCorrectPower:
    """Test correction of power by temperature coefficient."""

    def test_output_type_numeric(self):
        assert isinstance(calcparams.temp_correct_power(10, -0.37, 50), float)

    def test_output_type_series(self):
        assert isinstance(
            calcparams.temp_correct_power(pd.Series([10, 12, 15]), -0.37, 50), pd.Series
        )

    def test_high_temp_higher_power(self):
        power = 10
        corr_power = calcparams.temp_correct_power(power, -0.37, 50)
        assert corr_power > power

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

    def test_float_inputs(self):
        assert calcparams.back_of_module_temp(800, 30, 3) == pytest.approx(48.1671)

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix)
        temp_amb = pd.Series([26, 27, 27.5], index=ix)
        wind = pd.Series([0.5, 1, 2.5], index=ix)

        exp_results = pd.Series([48.0506544, 48.3709869, 46.6442104], index=ix)

        assert (
            pd.testing.assert_series_equal(
                calcparams.back_of_module_temp(poa, temp_amb, wind), exp_results
            )
            is None
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
    def test_float_inputs(self, capsys):
        cell_temp = calcparams.cell_temp(30, 850, verbose=False)
        captured = capsys.readouterr()
        assert cell_temp == pytest.approx(32.55)
        assert captured.out == ''

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix)
        temp_bom = pd.Series([26, 27, 27.5], index=ix)

        exp_results = pd.Series([28.415, 29.43, 29.936], index=ix)

        assert (
            pd.testing.assert_series_equal(calcparams.cell_temp(temp_bom, poa), exp_results)
            is None
        )

    @pytest.mark.parametrize(
        "racking, module_type, expected",
        [
            ("open_rack", "glass_cell_glass", 30.4),
            ("open_rack", "glass_cell_poly", 30.4),
            ("open_rack", "poly_tf_steel", 30.4),
            ("close_roof_mount", "glass_cell_glass", 28.8),
            ("insulated_back", "glass_cell_poly", 28),
        ],
    )
    def test_emp_heat_coeffs(self, racking, module_type, expected):
        bom = calcparams.cell_temp(28, 800, module_type=module_type, racking=racking)
        assert bom == pytest.approx(expected)

    def test_output_message_series_inputs(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix, name='poa')
        temp_bom = pd.Series([26, 27, 27.5], index=ix, name='bom_temp')
        calcparams.cell_temp(temp_bom, poa)
        
        # Get captured stdout
        captured = capsys.readouterr()
        stdout_content = captured.out
        
        print(stdout_content)
        assert stdout_content.rstrip('\n') == (
            'Calculating and adding "cell_temp" column using the Sandia temperature '
            'model assuming "glass_cell_poly" module type and "open_rack" racking '
            'from the "bom_temp" and "poa" columns.'
        )

    def test_output_message_float_inputs(self, capsys):
        poa = 805
        temp_bom = 26
        calcparams.cell_temp(temp_bom, poa)
        
        # Get captured stdout
        captured = capsys.readouterr()
        stdout_content = captured.out
        
        print(stdout_content)
        assert stdout_content.rstrip('\n') == (
            'Calculating and adding "cell_temp" column using the Sandia temperature '
            'model assuming "glass_cell_poly" module type and "open_rack" racking '
            'from the bom and poa values provided.'
        )
    
    def test_output_message_bom_series_poa_numeric(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        temp_bom = pd.Series([26, 27, 27.5], index=ix, name='bom_temp')
        poa = 805
        calcparams.cell_temp(temp_bom, poa)
        
        # Get captured stdout
        captured = capsys.readouterr()
        stdout_content = captured.out
        
        print(stdout_content)
        assert stdout_content.rstrip('\n') == (
            'Calculating and adding "cell_temp" column using the Sandia temperature '
            'model assuming "glass_cell_poly" module type and "open_rack" racking '
            'from the "bom_temp" column and poa value provided.'
        )

    def test_output_message_bom_numeric_poa_series(self, capsys):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([805, 810, 812], index=ix, name='poa')
        temp_bom = 26
        calcparams.cell_temp(temp_bom, poa)
        
        # Get captured stdout
        captured = capsys.readouterr()
        stdout_content = captured.out
        
        print(stdout_content)
        assert stdout_content.rstrip('\n') == (
            'Calculating and adding "cell_temp" column using the Sandia temperature '
            'model assuming "glass_cell_poly" module type and "open_rack" racking '
            'from the bom value provided and "poa" column.'
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
        poa = 100
        rear = 10
        assert calcparams.e_total(poa, rear) == 107
    
    def test_numeric_non_default_bifaciality(self):
        poa = 100
        rear = 10
        assert calcparams.e_total(poa, rear, bifaciality=0.5) == 105

    def test_numeric_non_default_bifi_frac(self):
        poa = 100
        rear = 10
        assert calcparams.e_total(poa, rear, bifaciality=1, bifacial_frac=0.5) == 105

    def test_numeric_non_default_bifaciality_and_bifacial_frac(self):
        poa = 100
        rear = 20
        assert calcparams.e_total(poa, rear, bifaciality=0.5, bifacial_frac=0.5) == 105

    def test_series_inputs(self):
        ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)
        poa = pd.Series([100, 110, 120], index=ix)
        rear = pd.Series([100, 150, 200], index=ix)

        exp_results = pd.Series([170, 215, 260], index=ix)

        pd.testing.assert_series_equal(
            calcparams.e_total(poa, rear), exp_results,
            check_dtype=False
        )