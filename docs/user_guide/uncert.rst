.. _uncert:

Test Uncertainty
================

This section discusses the approach to quantifying uncertainty of capacity test results 
implemented in pvcaptest.

Spatial Uncertainty
-------------------
This section discusses the calculation of spatial uncertainty. The equation for the
spatial uncertainty at each time interval is from ASTM PTC 19.1-2013 [1]_.

The ``CapData.spatial_uncert`` method calculates the spatial uncertainty for each of the 
independent regression variables defined in the regression equation. 

To obtain a meaningful spatial uncertainty there must be multiple sensors measuring the
same variable. The ``spatial_uncert`` method assumes that the ``agg_sensors`` method was
used to aggregate the measurements of the same variable from multiple sensors across
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