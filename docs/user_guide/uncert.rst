.. _uncert:

Test Uncertainty
================

This section discusses the approach to quantifying uncertainty implemented in pvcaptest.

Spatial Uncertainty
-------------------
This section discusses the calculation of spatial uncertainty. The approach and equations
used are derived from the ASTM PTC 

The ``CapData.spatial_uncert`` method calculates the spatial uncertainty for each of the 
independent regression variables defined in the regression equation. 

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
| :math:`\bar{X_{i}} =` average for the sampled measurand
| :math:`\bar{\bar{X}} =` grand average for all averaged measurands
| :math:`J =` number of sensors (i.e., spatial measurement locations)

The uncertainties calculated above for each time interval, :math:`b_{spatial,j}` are
combined to determine a spatial uncertainty across all time intervals.

.. math::

   b_{spatial} = \sqrt{\frac{\sum_{j=1}^{N}{b_{spatial, j}}^{2}}{N}}

where,

:math:`N` = number of averaging intervals in the final regression

