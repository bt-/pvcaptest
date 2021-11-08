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

class FilterBase(param.Parameterized):
    """I think this class could actually just be the new version of CapData, which will have the data parameters which is a DataFrame"""
    value = param.DataFrame(precedence=-1)

class FilterIrradiance(FilterBase):
    input = param.ClassSelector(class_=FilterBase, constant=True, precedence=-1)
    description = param.String(default='This filter removes irradiance below {} and above {}', precedence=-1)

    value = param.DataFrame(precedence=-1)

    x_column = param.String('POA pyranometer W/m2', precedence=-1)
    y_column = param.String('real power meter kW', precedence=-1)

    range = param.Range(default=(0,1200), bounds=(0, 1200))

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value()

    def __str__(self):
        return self.description.format(self.range[0], self.range[1])

    @param.depends('input.value', 'x_column', 'y_column', 'range', watch=True)
    def _update_value(self):
        input = self.input.value
        if input is None or input.empty:
            self.value = pd.DataFrame({self.x_column: [], self.y_column: []})
        else:
            min = self.range[0]
            max = self.range[1]
            x_column = self.x_column
            filter = (input[x_column] >= min) & (input[x_column] <= max)
            self.value = input[filter]

    @param.depends('value')
    def plot(self):
        return self.value.hvplot(kind='scatter', x=self.x_column, y=self.y_column).opts(xlim=self.param.range.bounds).opts(height=500, width=1000)

class FilterTime(FilterBase):
    input = param.ClassSelector(class_=FilterBase, constant=True, precedence=-1)
#     value = param.DataFrame(precedence=-1)
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

    @param.depends('input.value', 'period', watch=True)
    def _update_value(self):
        input = self.input.value
        if input is None or input.empty:
            self.value = pd.DataFrame({self.x_column: [], self.y_column: []})
        else:
            if self.drop:
                ix = input.loc[self.period[0]:self.period[1], :].index
                self.value = input.loc[input.index.difference(ix), :]
            else:
                self.value = input.loc[self.period[0]:self.period[1], :]

    @param.depends('value')
    def plot(self):
        return self.value.hvplot(kind='scatter', x='POA pyranometer W/m2', y='real power meter kW').relabel('Irr').opts(height=500, width=1000)

class FilterOutliers(FilterBase):
    input = param.ClassSelector(class_=FilterBase, constant=True, precedence=-1)
    value = param.DataFrame(precedence=-1)
    contamination = param.Number(default=0.04, bounds=(0, 1))
    support_fraction = param.Number(default=0.9, bounds=(0, 1))
    apply = param.Boolean(default=True)
    x_column = param.String('POA pyranometer W/m2', precedence=-1)
    y_column = param.String('real power meter kW', precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value()

    @param.depends('input.value', 'contamination', 'support_fraction', 'apply', watch=True)
    def _update_value(self):
        input = self.input.value
        if input is None or input.empty:
            self.value = pd.DataFrame({self.x_column: [], self.y_column: []})
        else:
            if self.apply:
                XandY = input[[self.x_column, self.y_column]].copy()
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
                self.value = input[clf_1.predict(X1) == 1]
            else:
                self.value = input

    @param.depends('value')
    def plot(self):
        return self.value.hvplot(kind='scatter', x=self.x_column, y=self.y_column).opts(height=500, width=1000)
