.. ptypes.rst
.. Copyright (c) 2013-2016 Pablo Acosta-Serafini
.. See LICENSE for details
.. py:module:: peng.ptypes

************
Pseudo-types
************

.. _EngineeringNotationNumber:

EngineeringNotationNumber
-------------------------

String with a number represented in engineering notation. Optional leading
whitespace can precede the mantissa; optional whitespace can also follow the
engineering suffix. An optional sign (+ or -) can precede the mantissa after
the leading whitespace. The suffix must be one of :code:`'y'`, :code:`'z'`,
:code:`'a'`, :code:`'f'`, :code:`'p'`, :code:`'n'`, :code:`'u'`, :code:`'m'`,
:code:`' '` (space), :code:`'k'`, :code:`'M'`, :code:`'G'`, :code:`'T'`,
:code:`'P'`, :code:`'E'`, :code:`'Z'` or :code:`'Y'`.  The correspondence
between suffix and floating point exponent is:

+----------+-------+--------+
| Exponent | Name  | Suffix |
+==========+=======+========+
| 1E-24    | yocto | y      |
+----------+-------+--------+
| 1E-21    | zepto | z      |
+----------+-------+--------+
| 1E-18    | atto  | a      |
+----------+-------+--------+
| 1E-15    | femto | f      |
+----------+-------+--------+
| 1E-12    | pico  | p      |
+----------+-------+--------+
| 1E-9     | nano  | n      |
+----------+-------+--------+
| 1E-6     | micro | u      |
+----------+-------+--------+
| 1E-3     | milli | m      |
+----------+-------+--------+
| 1E+0     |       |        |
+----------+-------+--------+
| 1E+3     | kilo  | k      |
+----------+-------+--------+
| 1E+6     | mega  | M      |
+----------+-------+--------+
| 1E+9     | giga  | G      |
+----------+-------+--------+
| 1E+12    | tera  | T      |
+----------+-------+--------+
| 1E+15    | peta  | P      |
+----------+-------+--------+
| 1E+18    | exa   | E      |
+----------+-------+--------+
| 1E+21    | zetta | Z      |
+----------+-------+--------+
| 1E+24    | yotta | Y      |
+----------+-------+--------+

.. _EngineeringNotationSuffix:

EngineeringNotationSuffix
-------------------------

A single character string, one  of :code:`'y'`, :code:`'z'`, :code:`'a'`,
:code:`'f'`, :code:`'p'`, :code:`'n'`, :code:`'u'`, :code:`'m'`,
:code:`' '` (space), :code:`'k'`, :code:`'M'`, :code:`'G'`, :code:`'T'`,
:code:`'P'`, :code:`'E'`, :code:`'Z'` or :code:`'Y'`.
:ref:`EngineeringNotationNumber` lists the correspondence between
suffix and floating point exponent

.. _IncreasingRealNumpyVector:

IncreasingRealNumpyVector
-------------------------
Numpy vector in which all elements are real (integers and/or floats) and
monotonically increasing (each element is strictly greater than the
preceding one)

.. _NumberNumpyVector:

NumberNumpyVector
-----------------
Numpy vector in which all elements are integers and/or floats and/or complex

.. _TouchstoneData:

TouchstoneData
--------------

A dictionary with the following structure:

* **points** (*integer*) -- Number of data points

* **freq** (:ref:`IncreasingRealNumpyVector`) -- Frequency vector

* **data** (:ref:`NumberNumpyVector`) -- Parameter data, its size is equal to
  :code:`nports` x :code:`nports` x :code:`points` where :code:`nports`
  represents the number of ports in the file

The dictionary keys are case sensitive

.. _TouchstoneNoiseData:

TouchstoneNoiseData
-------------------

A dictionary with the following structure:

* **points** (*integer*) -- Number of data points

* **freq** (:ref:`IncreasingRealNumpyVector`) -- Frequency vector

* **nf** (:ref:`RealNumpyVector`) -- Minimum noise figure vector in decibels

* **rc** (:ref:`NumberNumpyVector`) -- Source source reflection coefficient to
  realize minimum noise figure

* **res** (:ref:`RealNumpyVector`) -- Normalized effective noise resistance

The dictionary keys are case sensitive

.. _TouchstoneOptions:

TouchstoneOptions
-----------------

A dictionary with the following structure:

* **units** (*string*) -- Frequency units, one of :code:`'GHz'`, :code:`'MHz'`,
  :code:`'KHz'` or :code:`'Hz'` (case insensitive, default :code:`'GHz'`)

* **ptype** (*string*) -- Parameter type, one of :code:`'S'`, :code:`'Y'`,
  :code:`'Z'`, :code:`'H'` or :code:`'G'` (case insensitive, default
  :code:`'S'`)

* **pformat** (*string*) -- Data point format type, one of :code:`'DB'`
  (decibels), :code:`'MA'` (magnitude and angle), or :code:`'RI'` (real and
  imaginary) (case insensitive, default :code:`'MA'`)

* **z0** (float) -- Reference resistance in Ohms, default 50 Ohms

The dictionary keys are case sensitive

.. _WaveInterpOption:

WaveInterpOption
----------------
String representing a waveform interpolation type, one of :code:`'CONTINUOUS'`
or :code:`'STAIRCASE'` (case insensitive)

.. _WaveScaleOption:

WaveScaleOption
---------------
String representing a waveform scale type, one of :code:`'LINEAR'`
or :code:`'LOG'` (case insensitive)

.. _WaveVectors:

WaveVectors
-----------
Non-empty list of tuples in which each tuple is a waveform point. The first
item of a point tuple is its independent variable and the second item of a
point tuple is its dependent variable. The vector formed by the first item
of the point tuples is of :ref:`IncreasingRealNumpyVector` pseudo-type; the
vector formed by the second item of the point tuples is of
:ref:`RealNumpyVector` pseudo-type

*********
Contracts
*********

.. autofunction:: peng.ptypes.engineering_notation_number
.. autofunction:: peng.ptypes.engineering_notation_suffix
.. autofunction:: peng.ptypes.increasing_real_numpy_vector
.. autofunction:: peng.ptypes.number_numpy_vector
.. autofunction:: peng.ptypes.touchstone_data
.. autofunction:: peng.ptypes.touchstone_noise_data
.. autofunction:: peng.ptypes.touchstone_options
.. autofunction:: peng.ptypes.wave_interp_option
.. autofunction:: peng.ptypes.wave_scale_option
.. autofunction:: peng.ptypes.wave_vectors
