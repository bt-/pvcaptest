.. _uncert:

################
Test Uncertainty
################

This section discusses the approach to quantifying uncertainty of capacity test results 
implemented in pvcaptest.


Regression Uncertainty (Random Standard Uncertainty)
====================================================
Pvcaptest uses the `statsmodels <https://www.statsmodels.org/stable/index.html>`__ package
to perform the regression analysis. Once the regression is fitted the regression results
object provides a method to create a prediction (substitute the RCs into the model),
which also provides the "standard error of the prediction" (the ``se_obs`` attribute of 
the prediction object). This provides the Random Standard Uncertainty of the predicted 
value, :math:`S_{Y}`. The ``CapData.regression_uncertainty`` method calculates the 
regression uncertainty and stores it in the ``sy`` attribute of the ``CapData`` object
as a fraction of the predicted power.

Systematic Standard Uncertainty
===============================
The Systematic Standard Uncertainty, :math:`b_{Y}` will account for the uncertainty in 
the meterological measurements due to the instrument uncertainties and the spatial 
uncertainty due to the estimation of the average conditions of the PV plant by a few 
point measurements.

Instrument Uncertainty
----------------------
This section discusses the calculation of instrument uncertainty of the sensors. A capacity
test following the ASTM E2848 standard will rely on measurements of POA irradiance, ambient
temperature, and wind speed.

Pvcaptest requires that the user provide the instrument uncertainty for each group of 
sensors. The uncertainties may be relative or absolute. Absolute uncertainties are 
assumed to be valid at the reporting conditions. Relative uncertainties are used to 
calculate absolute uncertainties at the reporting conditions by the
``CapData.instrument_uncert`` method.

For example, for a reporting irradiance of 850 W/m\ :sup:`2` and a measurement uncertainty of 3%
the absolute uncertainty would be 25.5 W/m\ :sup:`2`.

The absolute instrument uncertainties, :math:`b_{inst}` are saved in the ``CapData.u_instrument``
attribute. These are combined with the spatial uncertainties, as described below.

Currently, pvcaptest does not provide functionality to calculate the instrument uncertainty.
Future work may include functionality to calculate the instrument uncertainty of irradiance
measurements.

Spatial Uncertainty
-------------------
This section discusses the calculation of spatial uncertainty. The equation for the
spatial uncertainty at each time interval is from ASTM PTC 19.1-2013 [1]_.

The ``CapData.spatial_uncert`` method calculates the spatial uncertainty for each of the 
independent regression variables defined in the regression equation. 

To obtain a meaningful spatial uncertainty there must be multiple sensors measuring the
same variable. The ``spatial_uncert`` method assumes that the ``agg_sensors`` method was
used to aggregate the measurements of the same variable across multiple sensors for
each time interval. Because the ``agg_sensors`` method modifies the ``regression_cols``
attribute it also saves a copy of the original ``regression_cols`` attribute to 
``pre_agg_reg_trans``. The ``spatial_uncert`` method uses uses the column groups
identified in ``pre_agg_reg_trans`` to calculate the spatial uncertainty for each of the
regression terms on the right side of the regression equation defined in
``regression_formula``.

E.g., for the standard regression equation:

::

    power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1

the spatial uncertainties associated with the ``poa``, ``t_amb``, and ``w_vel`` terms
are calculated. The following calculations are performed to calculate the spatial
uncertainty, :math:`b_{spatial}`, for each term.

First, for each time interval the spatial uncertainty is calculated as:

.. math::

   b_{spatial,j} = \frac{s_{spatial}}{\sqrt{J}}

where,

.. math::

   S_{spatial} = \sqrt{\frac{\sum_{i=1}^{J}{(\bar{X_{i}} - \bar{\bar{X}})}^{2}}{J - 1}}

and where,

| :math:`S_{spatial} =` standard deviation of the multiple sensor time-averaged values
| :math:`\bar{X_{i}} =` average for the sampled measurand e.g. if using 1 minute
   interval data the 1 minute average of the samples recorded during that minute
| :math:`\bar{\bar{X}} =` `grand average <https://en.wikipedia.org/wiki/Grand_mean>`__
   for all averaged measurands i.e the average of all :math:`\bar{X_{i}}`
| :math:`J =` number of sensors (i.e., spatial measurement locations)

The uncertainties calculated above for each time interval, :math:`b_{spatial,j}` are
combined to determine a spatial uncertainty across all time intervals.

.. math::

   b_{spatial} = \sqrt{\frac{\sum_{j=1}^{N}{b_{spatial, j}}^{2}}{N}}

where,

:math:`N` = number of averaging intervals in the final regression

References
----------
.. [1] "Test Uncertainty, Performance Test Codes", ASME PTC 19.1-2013, 8-4 Spatial
   Variations, pp 33, The American Society of Mechanical Engineers, New York, NY, 2013