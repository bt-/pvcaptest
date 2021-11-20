import datetime as dt

import pandas as pd
import numpy as np

import sklearn.covariance as sk_cv

import holoviews as hv
import panel as pn
import param
import hvplot.pandas

from captest import capdata as pvc

hv.extension('bokeh')
pn.extension()
pn.extension(sizing_mode="stretch_width")

class CapTest(param.Parameterized):
    steps = param.List(precedence=-1)
    step = param.ClassSelector(
        class_=pvc.CapData,
        is_instance=False,
        per_instance=False,
        instantiate=True,
    )
    step_widgets = param.List(precedence=-1)
    cd = param.ClassSelector(class_=pvc.CapData, constant=False, precedence=-1)

    def __init__(self, cd, **params):
        super().__init__(**params)
        self.cd = cd

    def add_step(self, event):
        if len(self.steps) == 0:
            input_step = self.cd
        else:
            input_step = self.steps[-1]

        if self.step.name == 'FilterTime':
            step = self.step(
                input=input_step,
                period=(self.cd.data.index[0], self.cd.data.index[-1]),
                poa_column = self.cd.poa_column,
                power_column = self.cd.power_column,
            )
        elif self.step.name == 'FilterIrradiance':
            step = self.step(
                input=input_step,
                range=(
                    self.cd.data[self.cd.poa_column].min(),
                    self.cd.data[self.cd.poa_column].max()
                ),
                poa_column = self.cd.poa_column,
                power_column = self.cd.power_column,
            )
        elif self.step.name == 'FilterClearSky':
            step = self.step(
                input=input_step,
                # will need way to id ghi col without hard coding
                ghi_col='irr-ghi-mean-agg',
                poa_column = self.cd.poa_column,
                power_column = self.cd.power_column,
            )
        else:
            step = self.step(
                input=input_step,
                poa_column = self.cd.poa_column,
                power_column = self.cd.power_column,
            )

        if isinstance(step, FilterIrradiance):
            filter_control = pn.Param(
                step, widgets={
                    'range': pn.widgets.EditableRangeSlider,
                }
            )
            self.step_widgets.append(filter_control)
        elif isinstance(step, FilterTime):
            filter_control = pn.Param(
                step, widgets={
                    'period': pn.widgets.DatetimeRangePicker,
                }
            )
            self.step_widgets.append(filter_control)
        else:
            self.step_widgets.append(step.param)

        self.steps.append(step)

    def reset_instance_list(self, event):
        self.steps = []
        self.step_widgets = []


class FilterIrradiance(pvc.CapData):
    input = param.ClassSelector(class_=pvc.CapData, constant=True, precedence=-1)
    poa_column = param.String(precedence=-1)
    power_column = param.String(precedence=-1)
    description = param.String(
        default='This filter removes irradiance below {} and above {}',
        precedence=-1
    )

    range = param.Range(default=(0,1200), bounds=(-10, 1400))

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value()

    def __str__(self):
        return self.description.format(self.range[0], self.range[1])

    @param.depends('input.data', 'poa_column', 'power_column', 'range', watch=True)
    def _update_value(self):
        input = self.input.data
        if input is None or input.empty:
            self.data = pd.DataFrame({self.poa_column: [], self.power_column: []})
        else:
            self.data = input[
                (input[self.poa_column] >= self.range[0]) &
                (input[self.poa_column] <= self.range[1])
            ]

class FilterTime(pvc.CapData):
    input = param.ClassSelector(class_=pvc.CapData, constant=True, precedence=-1)
    description = param.String(default='This filter {} the data from {} to {}', precedence=-1)
    period = param.CalendarDateRange(default=(dt.date(2019, 11, 7), dt.date(2020, 1, 22)))
    drop = param.Boolean(default=False)
    poa_column = param.String(precedence=-1)
    power_column = param.String(precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value()

    def __str__(self):
        if self.drop:
            filtering_action = 'removes'
        else:
            filtering_action = 'keeps only'
        return self.description.format(filtering_action, self.period[0], self.period[1])

    @param.depends('input.data', 'period', watch=True)
    def _update_value(self):
        input = self.input.data
        if input is None or input.empty:
            self.data = pd.DataFrame({self.poa_column: [], self.power_column: []})
        else:
            if self.drop:
                ix = input.loc[self.period[0]:self.period[1], :].index
                self.data = input.loc[input.index.difference(ix), :]
            else:
                self.data = input.loc[self.period[0]:self.period[1], :]

class FilterOutliers(pvc.CapData):
    input = param.ClassSelector(class_=pvc.CapData, constant=True, precedence=-1)
    contamination = param.Number(default=0.04, bounds=(0, 1))
    support_fraction = param.Number(default=0.9, bounds=(0, 1))
    poa_column = param.String(precedence=-1)
    power_column = param.String(precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value()

    @param.depends('input.data', 'contamination', 'support_fraction', watch=True)
    def _update_value(self):
        input = self.input.data
        if input is None or input.empty:
            self.data = pd.DataFrame({self.poa_column: [], self.power_column: []})
        else:
            XandY = input[[self.poa_column, self.power_column]].copy()
            if XandY.shape[1] > 2:
                return warnings.warn('Too many columns. Try running '
                                     'aggregate_sensors before using '
                                     'filter_outliers.')
            X1 = XandY.values

            kwargs = {
                'support_fraction': self.support_fraction,
                'contamination': self.contamination,
            }

            clf_1 = sk_cv.EllipticEnvelope(**kwargs)
            clf_1.fit(X1)
            self.data = input[clf_1.predict(X1) == 1]

class FilterClearSky(pvc.CapData):
    input = param.ClassSelector(class_=pvc.CapData, constant=True, precedence=-1)
    poa_column = param.String(precedence=-1)
    power_column = param.String(precedence=-1)
    window_length = param.Integer(default=10)
    ghi_col = param.String(default=None)
    keep_clear = param.Boolean(default=True)

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value()

    @param.depends('input.data', 'window_length', 'ghi_col', 'keep_clear', watch=True)
    def _update_value(self):
        input = self.input.data
        if input is None or input.empty:
            self.data = pd.DataFrame({self.poa_column: [], self.power_column: []})
        else:
            clear_per = pvc.detect_clearsky(
                input[self.ghi_col],
                input['ghi_mod_csky'],
                input[self.ghi_col].index,
                self.window_length,
            )
            if not any(clear_per):
                return warnings.warn('No clear periods detected. Try increasing '
                                     'the window length.')

            if self.keep_clear:
                self.data = input[clear_per]
            else:
                self.data = input[~clear_per]
