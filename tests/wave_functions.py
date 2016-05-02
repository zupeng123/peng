# wave_functions.py
# Copyright (c) 2013-2016 Pablo Acosta-Serafini
# See LICENSE for details
# pylint: disable=C0111

# Standard library imports
import copy
import math
# PyPI imports
import numpy
import pytest
from pmisc import AE, AI
# Intra-package imports
import peng
from .support import cmp_vectors, std_obj


###
# Helper functions
###
def strict_compare_waves(dep_vector=None, rfunc=None, rdesc=None,
    dep_units=None, nobj=None, indep_vector=None):
    """ Strictly compare waveform objects """
    # pylint: disable=R0913
    obj = rfunc(
        std_obj(
            indep_name='obj', indep_vector=indep_vector, dep_vector=dep_vector
        )
    )
    ref = std_obj(
        indep_name='{0}(obj)'.format(rdesc),
        indep_vector=indep_vector,
        dep_vector=(
            copy.copy(obj.dep_vector)
            if dep_vector is None else (
                nobj if isinstance(nobj, numpy.ndarray) else nobj(dep_vector)
            )
        ),
        dep_units=dep_units
    )
    assert obj == ref
    assert obj.indep_name == ref.indep_name


###
# Test functions
###
def test_acos():
    """ Test acos function behavior """
    dep_vector = numpy.array([0.1, 0.98, 0.5])
    strict_compare_waves(
        dep_vector, peng.acos, 'acos', 'rad', numpy.arccos
    )


@pytest.mark.wave_functions
def test_acos_exceptions():
    """ Test acos function exceptions """
    dep_vector = numpy.array([-1.01, 0.98, 0.5])
    obj_a = std_obj(indep_name='obj_a', dep_vector=dep_vector)
    dep_vector = numpy.array([1.01, 0.98, 0.5])
    obj_b = std_obj(indep_name='obj_b', dep_vector=dep_vector)
    for item in [obj_a, obj_b]:
        AE(peng.acos, ValueError, 'Math domain error', item)


def test_acosh():
    """ Test acosh function behavior """
    dep_vector = numpy.array([1.1, 1.98, 1.5])
    strict_compare_waves(
        dep_vector, peng.acosh, 'acosh', '', numpy.arccosh
    )


@pytest.mark.wave_functions
def test_acosh_exceptions():
    """ Test acosh function exceptions """
    dep_vector = numpy.array([0.99, 0.98, 0.5])
    obj = std_obj(indep_name='obj_a', dep_vector=dep_vector)
    AE(peng.acosh, ValueError, 'Math domain error', obj)


def test_asin():
    """ Test asin function behavior """
    dep_vector = numpy.array([0.1, 0.98, 0.5])
    strict_compare_waves(
        dep_vector, peng.asin, 'asin', 'rad', numpy.arcsin
    )


@pytest.mark.wave_functions
def test_asin_exceptions():
    """ Test asin function exceptions """
    dep_vector = numpy.array([-1.01, 0.98, 0.5])
    obj_a = std_obj(indep_name='obj_a', dep_vector=dep_vector)
    dep_vector = numpy.array([1.01, 0.98, 0.5])
    obj_b = std_obj(indep_name='obj_b', dep_vector=dep_vector)
    for item in [obj_a, obj_b]:
        AE(peng.asin, ValueError, 'Math domain error', item)


def test_asinh():
    """ Test asinh function behavior """
    dep_vector = numpy.array([1.1, 1.98, 1.5])
    strict_compare_waves(
        dep_vector, peng.asinh, 'asinh', '', numpy.arcsinh
    )


def test_atan():
    """ Test atan function behavior """
    dep_vector = numpy.array([1.1, 1.98, 1.5])
    strict_compare_waves(
        dep_vector, peng.atan, 'atan', 'rad', numpy.arctan
    )


def test_atanh():
    """ Test atanh function behavior """
    dep_vector = numpy.array([0.1, 0.98, 0.5])
    strict_compare_waves(
        dep_vector, peng.atanh, 'atanh', '', numpy.arctanh
    )


@pytest.mark.wave_functions
def test_atanh_exceptions():
    """ Test atanh function exceptions """
    dep_vector = numpy.array([-1.01, 0.98, 0.5])
    obj_a = std_obj(indep_name='obj_a', dep_vector=dep_vector)
    dep_vector = numpy.array([1.01, 0.98, 0.5])
    obj_b = std_obj(indep_name='obj_b', dep_vector=dep_vector)
    for item in [obj_a, obj_b]:
        AE(peng.atanh, ValueError, 'Math domain error', item)


def test_average():
    """ Test average and naverage functions behavior """
    obj = std_obj(
        indep_vector=numpy.array([1, 2, 3, 7, 9, 15]),
        dep_vector=numpy.array([6, 5, 4, 8.2, 7, 7.25]),
        indep_name='obj',
        interp='CONTINUOUS',
        indep_scale='LINEAR'
    )
    act = peng.average(obj)
    ref = std_obj(
        indep_name='average(obj)',
        indep_vector=numpy.array([1, 2, 3, 7, 9, 15]),
        dep_vector=numpy.array(
            [6.0, 5.5, 5.0, 5.73333333333, 6.2, 6.59642857143]
        ),
        dep_units='Volts',
        interp='CONTINUOUS',
        indep_scale='LINEAR'
    )
    assert act == ref
    obj = std_obj(indep_name='obj', interp='CONTINUOUS', indep_scale='LINEAR')
    act = peng.average(obj)
    ref = std_obj(
        indep_name='average(obj)',
        dep_vector=numpy.array([6.0, 5.5, 5.0]),
        dep_units='Volts',
        interp='CONTINUOUS',
        indep_scale='LINEAR'
    )
    assert act.indep_name == ref.indep_name
    act = peng.average(obj, indep_min=1.5)
    cmp_vectors(act.dep_vector, numpy.array([5.5, 5.25, 4.75]))
    assert peng.naverage(obj, indep_min=1.5) == act[-1].dep_var
    act = peng.average(obj, indep_max=2.5)
    cmp_vectors(act.dep_vector, numpy.array([6, 5.5, 5.25]))
    assert peng.naverage(obj, indep_max=2.5) == act[-1].dep_var
    act = peng.average(obj, indep_min=1.5, indep_max=2.5)
    cmp_vectors(act.dep_vector, numpy.array([5.5, 5.25, 5.0]))
    num = peng.naverage(obj, indep_min=1.5, indep_max=2.5)
    assert num == act[-1].dep_var


@pytest.mark.wave_functions
def test_bound_exceptions():
    """ Test exceptions of functions that have range bounding """
    obj = std_obj(indep_name='obj')
    items = [
        peng.average, peng.derivative, peng.integral,
        peng.naverage, peng.nintegral, peng.nmax,
        peng.nmin,
    ]
    for item in items:
        AI(item, 'indep_min', wave=obj, indep_min='a')
        AI(item, 'indep_max', wave=obj, indep_max='a')
        msg = 'Incongruent `indep_min` and `indep_max` arguments'
        AE(item, RuntimeError, msg, wave=obj, indep_min=1.5, indep_max=1)
        AI(item, 'indep_min', wave=obj, indep_min=0)
        AI(item, 'indep_max', wave=obj, indep_max=10)


def test_ceil():
    """ Test ceil function behavior """
    dep_vector = numpy.array([10.41, 1.98, 1.0])
    strict_compare_waves(
        dep_vector, peng.ceil, 'ceil', 'Volts', numpy.ceil
    )


def test_cos():
    """ Test cos function behavior """
    dep_vector = numpy.array([1.1, 1.98, 1.5])
    strict_compare_waves(
        dep_vector, peng.cos, 'cos', '', numpy.cos
    )


def test_cosh():
    """ Test cosh function behavior """
    dep_vector = numpy.array([0.1, 0.98, 0.5])
    strict_compare_waves(
        dep_vector, peng.cosh, 'cosh', '', numpy.cosh
    )


def test_db():
    """ Test db function behavior """
    dep_vector = numpy.array([10, 100, 1000])
    strict_compare_waves(
        dep_vector, peng.db, 'db', 'dB', numpy.array([20.0, 40.0, 60.0])
    )


@pytest.mark.wave_functions
def test_db_exceptions():
    """ Test db function exceptions """
    dep_vector = numpy.array([0, 0.98, 0.5])
    obj = std_obj(indep_name='obj', dep_vector=dep_vector)
    AE(peng.db, ValueError, 'Math domain error', obj)


def test_derivative():
    """ Test derivative method behavior """
    indep_vector = numpy.array([1, 2.5, 3, 4.75])
    dep_vector = numpy.array([6, 5, 5.5, 1])
    strict_compare_waves(
        dep_vector,
        peng.derivative,
        'derivative',
        'Volts/Sec',
        numpy.array(
            [
                -0.66666666666,
                -0.66666666666,
                1.0,
                -2.5714285714
            ]
        ),
        indep_vector,
    )
    obj = std_obj('obj', indep_vector, dep_vector)
    obj.indep_units = ''
    assert peng.derivative(obj).dep_units == 'Volts'
    obj.indep_units = 'Sec'
    obj.dep_units = ''
    assert peng.derivative(obj).dep_units == '1/Sec'
    obj.indep_units = ''
    obj.dep_units = ''
    assert peng.derivative(obj).dep_units == ''


def test_exp():
    """ Test exp function behavior """
    strict_compare_waves(None, peng.exp, 'exp', '', numpy.exp)


def test_floor():
    """ Test floor function behavior """
    dep_vector = numpy.array([10.41, 1.98, 1.0])
    strict_compare_waves(
        dep_vector, peng.floor, 'floor', 'Volts', numpy.floor
    )


@pytest.mark.wave_functions
def test_funcs_exceptions():
    """ Test argument wave not valid exception """
    items = [
        peng.acos, peng.acosh, peng.asin, peng.asinh,
        peng.atan, peng.atanh, peng.average, peng.ceil,
        peng.cos, peng.cosh, peng.db, peng.derivative,
        peng.exp, peng.floor, peng.imag, peng.integral,
        peng.log, peng.log10, peng.naverage,
        peng.nintegral, peng.nmax, peng.nmin, peng.phase,
        peng.real, peng.round, peng.sin, peng.sinh,
        peng.sqrt, peng.tan, peng.tanh, peng.wcomplex,
        peng.wfloat, peng.wint,
    ]
    for item in items:
        AI(item, 'wave', 'a')


def test_imag():
    """ Test imag function behavior """
    dep_vector = numpy.array([10.41, 1+3j, 10-0.5j])
    strict_compare_waves(
        dep_vector, peng.imag, 'imag', 'Volts', numpy.imag
    )


def test_integral():
    """ Test integral and nintegral methods behavior """
    indep_vector = numpy.array([1, 2.5, 3, 4.75])
    dep_vector = numpy.array([6, 5, 5.5, 1])
    strict_compare_waves(
        dep_vector,
        peng.integral,
        'integral',
        'Volts*Sec',
        numpy.array([0.0, 8.25, 10.875, 16.5625]),
        indep_vector
    )
    obj = std_obj('obj', indep_vector, dep_vector)
    act = peng.integral(obj)
    cmp_vectors(
        numpy.array([act.dep_vector[-1]]),
        numpy.array([peng.nintegral(obj)])
    )
    obj = std_obj('obj', indep_vector, dep_vector)
    obj.indep_units = ''
    assert peng.integral(obj).dep_units == 'Volts'
    obj.indep_units = 'Sec'
    obj.dep_units = ''
    assert peng.integral(obj).dep_units == 'Sec'
    obj.indep_units = ''
    obj.dep_units = ''
    assert peng.integral(obj).dep_units == ''


def test_log():
    """ Test log function behavior """
    dep_vector = numpy.array([10.41, 1.98, 1.0])
    strict_compare_waves(
        dep_vector, peng.log, 'log', '', numpy.log
    )


def test_log10():
    """ Test log function behavior """
    dep_vector = numpy.array([10.41, 1.98, 1.0])
    strict_compare_waves(
        dep_vector, peng.log10, 'log10', '', numpy.log10
    )


@pytest.mark.wave_functions
def test_log_exceptions():
    """ Test log and log10 function exceptions """
    dep_vector = numpy.array([0, 0.98, 0.5])
    obj = std_obj(indep_name='obj', dep_vector=dep_vector)
    items = [peng.log, peng.log10]
    for item in items:
        AE(item, ValueError, 'Math domain error', obj)


def test_nmax():
    """ Test nmax method behavior """
    obj = std_obj('obj', indep_scale='LINEAR', interp='CONTINUOUS')
    assert peng.nmax(obj) == 6
    assert peng.nmax(obj, 1.5) == 5.5


def test_nmin():
    """ Test nmax method behavior """
    obj = std_obj('obj', indep_scale='LINEAR', interp='CONTINUOUS')
    assert peng.nmin(obj) == 4
    assert peng.nmin(obj, indep_max=2.5) == 4.5


def test_phase():
    """ Test phase function behavior """
    indep_vector = numpy.arange(1, 12, 1)
    dep_vector = numpy.exp(complex(0, 1)*math.pi*numpy.arange(0.25, 3, 0.25))
    obj = peng.phase(
        std_obj(
            indep_name='obj', indep_vector=indep_vector, dep_vector=dep_vector
        ),
        unwrap=False
    )
    ref = math.pi*numpy.array(
        [0.25, 0.5, 0.75, 1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    )
    cmp_vectors(obj.indep_vector, indep_vector)
    cmp_vectors(obj.dep_vector, ref)
    assert obj.indep_name == 'phase(obj)'
    assert obj.dep_units == 'rad'
    obj = peng.phase(
        std_obj(
            indep_name='obj', indep_vector=indep_vector, dep_vector=dep_vector
        ),
    )
    ref = math.pi*numpy.arange(0.25, 3, 0.25)
    cmp_vectors(obj.dep_vector, ref)
    obj = peng.phase(
        std_obj(
            indep_name='obj', indep_vector=indep_vector, dep_vector=dep_vector
        ),
        unwrap=True
    )
    cmp_vectors(obj.dep_vector, ref)
    obj = peng.phase(
        std_obj(
            indep_name='obj', indep_vector=indep_vector, dep_vector=dep_vector
        ),
        unwrap=True,
        rad=False
    )
    cmp_vectors(obj.dep_vector, numpy.arange(45, 534, 45))


@pytest.mark.wave_functions
def test_phase_exceptions():
    """ Test phase function exceptions """
    obj = std_obj('obj')
    items = ['a', 5.0, (1, 2)]
    for item in items:
        AI(peng.phase, 'unwrap', wave=obj, unwrap=item)
        AI(peng.phase, 'rad', wave=obj, rad=item)


def test_real():
    """ Test real function behavior """
    dep_vector = numpy.array([10.41, 1+3j, 10-0.5j])
    strict_compare_waves(
        dep_vector, peng.real, 'real', 'Volts', numpy.real
    )


def test_round():
    """ Test wround method behavior """
    obj_a = std_obj('obj', dep_vector=numpy.array([5.4, 1.6, 0]))
    ref = std_obj('round(obj, 0)', dep_vector=numpy.array([5, 2, 0]))
    act = peng.round(obj_a)
    assert ref == act
    assert ref.indep_name == act.indep_name
    obj_b = std_obj('obj', dep_vector=numpy.array([5.47, 1.61, 0]))
    ref = std_obj('round(obj, 1)', dep_vector=numpy.array([5.5, 1.6, 0.0]))
    act = peng.round(obj_b, 1)
    assert ref == act
    assert ref.indep_name == act.indep_name


@pytest.mark.wave_functions
def test_round_exceptions():
    """ Test wround function exceptions """
    obj = std_obj(indep_name='obj')
    items = [-1, -0.0001, 'a', 3.5]
    for item in items:
        AI(peng.round, 'decimals', wave=obj, decimals=item)


def test_sin():
    """ Test sin function behavior """
    dep_vector = numpy.array([1.1, 1.98, 1.5])
    strict_compare_waves(
        dep_vector, peng.sin, 'sin', '', numpy.sin
    )


def test_sinh():
    """ Test sinh function behavior """
    dep_vector = numpy.array([0.1, 0.98, 0.5])
    strict_compare_waves(
        dep_vector, peng.sinh, 'sinh', '', numpy.sinh
    )



def test_sqrt():
    """ Test sqrt function behavior """
    dep_vector = numpy.array([1.1, 1.98, 1.5])
    strict_compare_waves(
        dep_vector, peng.sqrt, 'sqrt', 'Volts**0.5', numpy.sqrt
    )


def test_tan():
    """ Test tan function behavior """
    dep_vector = numpy.array([1.1, 1.98, 1.5])
    strict_compare_waves(
        dep_vector, peng.tan, 'tan', '', numpy.tan
    )


def test_tanh():
    """ Test tanh function behavior """
    dep_vector = numpy.array([0.1, 0.98, 0.5])
    strict_compare_waves(
        dep_vector, peng.tanh, 'tanh', '', numpy.tanh
    )


def test_wcomplex():
    """ Test wcomplex method behavior """
    obj_a = std_obj(
        indep_name='obj_a',
        dep_vector=numpy.array([3, 4, 5]),
    )
    obj_b = peng.wcomplex(obj_a)
    ref = std_obj(
        indep_name='ref',
        dep_vector=numpy.array([3+0j, 4+0j, 5+0j]),
    )
    assert obj_b == ref
    assert obj_b.dep_vector.dtype.name.startswith('complex')


def test_wfloat():
    """ Test wfloat method behavior """
    obj_a = std_obj('obj_a')
    obj_b = peng.wfloat(obj_a)
    ref = std_obj(
        indep_name='ref',
        dep_vector=numpy.array([6.0, 5.0, 4.0]),
    )
    assert obj_b == ref
    assert obj_b.dep_vector.dtype.name.startswith('float')


@pytest.mark.wave_functions
def test_wfloat_exceptions():
    """ Test wfloat function exceptions """
    dep_vector = numpy.array([0.99, 1+3j, 0.5])
    obj = std_obj(indep_name='obj_a', dep_vector=dep_vector)
    AE(peng.wfloat, TypeError, 'Cannot convert complex to float', obj)


def test_wint():
    """ Test wint method behavior """
    obj_a = std_obj('obj_a', dep_vector=numpy.array([5.5, 1.3, 3.4]))
    obj_b = peng.wint(obj_a)
    ref = std_obj(
        indep_name='ref', dep_vector=numpy.array([5, 1, 3]),
    )
    assert obj_b == ref
    assert obj_b.dep_vector.dtype.name.startswith('int')


@pytest.mark.wave_functions
def test_wint_exceptions():
    """ Test wint function exceptions """
    dep_vector = numpy.array([0.99, 1+3j, 0.5])
    obj = std_obj(indep_name='obj_a', dep_vector=dep_vector)
    AE(peng.wint, TypeError, 'Cannot convert complex to integer', obj)


def test_wvalue():
    """ Test wvalue method behavior """
    obj = std_obj('obj')
    assert peng.wvalue(obj, 0.9999999999999) == 6
    assert peng.wvalue(obj, 1.0) == 6
    assert peng.wvalue(obj, 1.0000000000001) == 6
    assert peng.wvalue(obj, 2.9999999999999) == 4
    assert peng.wvalue(obj, 3) == 4
    assert peng.wvalue(obj, 3.0000000000001) == 4
    assert peng.wvalue(obj, 1.5) == 5.5
    assert peng.wvalue(obj, 1.25) == 5.75
    assert peng.wvalue(obj, 2.5) == 4.5
    assert peng.wvalue(obj, 2.9) == 4.1


@pytest.mark.wave_functions
def test_wvalue_exceptions():
    """ Test wvalue function exceptions """
    AI(peng.wvalue, 'wave', 'a', 5)
    obj = std_obj(indep_name='obj')
    exmsg = (
        'Argument `indep_var` is not in the independent variable vector range'
    )
    AE(peng.wvalue, ValueError, exmsg, obj, 0)
    AE(peng.wvalue, ValueError, exmsg, obj, 0.999)
    AE(peng.wvalue, ValueError, exmsg, obj, 3.001)
    AE(peng.wvalue, ValueError, exmsg, obj, 4)
