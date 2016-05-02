.. api.rst
.. Copyright (c) 2013-2016 Pablo Acosta-Serafini
.. See LICENSE for details
.. py:module:: putil.eng

****************
Global variables
****************

.. autodata:: peng.FP_ATOL
.. autodata:: peng.FP_RTOL

************
Named tuples
************

.. autofunction:: peng.functions.EngPower
.. autofunction:: peng.NumComp
.. autofunction:: peng.wave_core.Point

*********
Functions
*********

.. autofunction:: peng.functions.no_exp
.. autofunction:: peng.functions.peng
.. autofunction:: peng.functions.peng_float
.. autofunction:: peng.functions.peng_frac
.. autofunction:: peng.functions.peng_int
.. autofunction:: peng.functions.peng_mant
.. autofunction:: peng.functions.peng_power
.. autofunction:: peng.functions.peng_suffix
.. autofunction:: peng.functions.peng_suffix_math
.. autofunction:: peng.functions.pprint_vector
.. autofunction:: peng.touchstone.read_touchstone
.. autofunction:: peng.touchstone.write_touchstone
.. autofunction:: peng.functions.remove_extra_delims
.. autofunction:: peng.functions.round_mantissa
.. autofunction:: peng.functions.to_scientific_string
.. autofunction:: peng.functions.to_scientific_tuple


********
Waveform
********

.. autofunction:: peng.wave_functions.acos
.. autofunction:: peng.wave_functions.acosh
.. autofunction:: peng.wave_functions.asin
.. autofunction:: peng.wave_functions.asinh
.. autofunction:: peng.wave_functions.atan
.. autofunction:: peng.wave_functions.atanh
.. autofunction:: peng.wave_functions.average
.. autofunction:: peng.wave_functions.ceil
.. autofunction:: peng.wave_functions.cos
.. autofunction:: peng.wave_functions.cosh
.. autofunction:: peng.wave_functions.db
.. autofunction:: peng.wave_functions.derivative
.. autofunction:: peng.wave_functions.exp
.. autofunction:: peng.wave_functions.floor
.. autofunction:: peng.wave_functions.imag
.. autofunction:: peng.wave_functions.integral
.. autofunction:: peng.wave_functions.log
.. autofunction:: peng.wave_functions.log10
.. autofunction:: peng.wave_functions.naverage
.. autofunction:: peng.wave_functions.nintegral
.. autofunction:: peng.wave_functions.nmax
.. autofunction:: peng.wave_functions.nmin
.. autofunction:: peng.wave_functions.phase
.. autofunction:: peng.wave_functions.real
.. autofunction:: peng.wave_functions.round
.. autofunction:: peng.wave_functions.sin
.. autofunction:: peng.wave_functions.sinh
.. autofunction:: peng.wave_functions.sqrt
.. autofunction:: peng.wave_functions.tan
.. autofunction:: peng.wave_functions.tanh
.. autofunction:: peng.wave_functions.wcomplex
.. autofunction:: peng.wave_functions.wfloat
.. autofunction:: peng.wave_functions.wint
.. autofunction:: peng.wave_functions.wvalue

.. autoclass:: peng.wave_core.Waveform
	:members: __add__,
                  __and__,
                  __bool__,
                  __contains__,
                  __copy__,
                  __delitem__,
                  __div__,
                  __eq__,
                  __floordiv__,
                  __ge__,
                  __getitem__,
                  __gt__,
                  __invert__,
                  __iter__,
                  __le__,
                  __len__,
                  __lshift__,
                  __lt__,
                  __mod__,
                  __mul__,
                  __ne__,
                  __neg__,
                  __nonzero__,
                  __or__,
                  __pos__,
                  __pow__,
                  __sub__,
                  __radd__,
                  __rand__,
                  __repr__,
                  __rfloordiv__,
                  __rlshift__,
                  __rmod__,
                  __rmul__,
                  __ror__,
                  __rpow__,
                  __rrshift__,
                  __rshift__,
                  __rsub__,
                  __rtruediv__,
                  __rxor__,
                  __setitem__,
                  __str__,
                  __truediv__,
                  __xor__,
                  complex,
                  dep_scale,
                  dep_units,
                  dep_vector,
                  indep_scale,
                  indep_units,
                  indep_vector,
                  interp,
                  real,
                  vectors
	:show-inheritance:
