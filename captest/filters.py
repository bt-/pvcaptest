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

# class FilterBase(param.Parameterized):
#     """I think this class could actually just be the new version of CapData, which will have the data parameters which is a DataFrame"""
#     value = param.DataFrame(precedence=-1)
#
#     poa_column = param.String('POA pyranometer W/m2', precedence=-1)
#     power_column = param.String('real power meter kW', precedence=-1)
#
#     @param.depends('value')
#     def plot(self):
#         return self.value.hvplot(kind='scatter', x=self.poa_column, y=self.power_column).opts(height=500, width=1000)


class FilterIrradiance(pvc.CapData):
    input = param.ClassSelector(class_=pvc.CapData, constant=True, precedence=-1)
    description = param.String(
        default='This filter removes irradiance below {} and above {}',
        precedence=-1
    )

    range = param.Range(default=(0,1200), bounds=(0, 1200))

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
    apply = param.Boolean(default=True)
    # poa_column = param.String('POA pyranometer W/m2', precedence=-1)
    # power_column = param.String('real power meter kW', precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value()

    @param.depends('input.data', 'contamination', 'support_fraction', 'apply', watch=True)
    def _update_value(self):
        input = self.input.data
        if input is None or input.empty:
            self.data = pd.DataFrame({self.poa_column: [], self.power_column: []})
        else:
            if self.apply:
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
            else:
                self.data = input
