import thermofun as thermofun
import pytest as pytest
import unittest

# Regression coverage for the memoization cache in ThermoFun/OptimizationUtils.h,
# exercised through the public Python API (which is the same path GEMS3K uses via
# TNode::load_all_thermodynamic_from_thermo - see CLAUDE.md).
#
# ThermoEngine's memoized property functions take pressure as an in-out C++
# reference: P == 0 means "at the saturation pressure", and the solvent model
# writes the computed Psat back through it. pybind11 cannot reflect that
# write-back to a Python float, so these tests cannot observe P directly (that
# half is covered by tests/memoization/src/main.cpp instead). What they can and
# do check is the symptom that made the underlying bug into a real-world
# incident: with the cache keyed on a dangling reference, a substance's
# properties could come back as another substance's - most dramatically, every
# entry from one pass collapsing onto the last one computed. So every symbol
# queried here must keep returning its own value, repeatably, regardless of what
# else was queried in between.


class TestMemoization(unittest.TestCase):

    def setUp(self):
        self.engine = thermofun.ThermoEngine('pytests/test-thermoengine-thermofun.json')
        self.engine2 = thermofun.ThermoEngine('pytests/test-aq17-gem-lma-thermofun.json')

    def test_saturation_pressure_substance_properties_stay_distinct_and_stable(self):
        T = 473.15
        # symbol -> gibbs_energy at P = 0 (saturation pressure), pinned against a
        # known-good build of the fixed OptimizationUtils.h
        expected = {
            'Ca+2':    -541308.716668614,
            'Mg+2':    -431145.4445186801,
            'Na+':     -268429.12018646183,
            'K+':      -295165.7631867845,
            'Cl-':     -136074.51529469137,
            'HCO3-':   -602350.0526394788,
            'CO3-2':   -506814.1330662342,
            'H4SiO4@': -1349526.3674365964,
        }

        # Pass 1, in dict order.
        pass1 = {sym: self.engine2.thermoPropertiesSubstance(T, 0, sym).gibbs_energy.val
                 for sym in expected}

        # Pass 2, reversed - a different sequence of preceding calls is exactly
        # what let a dangling-reference cache key return a neighbour's value.
        pass2 = {sym: self.engine2.thermoPropertiesSubstance(T, 0, sym).gibbs_energy.val
                 for sym in reversed(list(expected))}

        for sym, want in expected.items():
            assert pass1[sym] == pytest.approx(want, 1e-5, 1e-14)
            assert pass2[sym] == pytest.approx(want, 1e-5, 1e-14)

        # No two distinct substances collapsed onto the same cached entry.
        values = list(pass1.values())
        assert len(set(values)) == len(values)

    def test_saturation_pressure_reaction_properties_stay_distinct_and_stable(self):
        T = 298.15
        # symbol -> log_equilibrium_constant at P = 0, same rationale as above
        # but for ThermoPropertiesReactionFunction, the fourth memoized function.
        expected = {
            'Meionite-Ca':     80.87391613785806,
            'Gedrite-Mg':      86.35257372406875,
            'Tschermakite-Mg': 80.78800987679564,
            'Pargasite-Mg':    88.8475233660601,
            'Pyrope':          58.20122833383338,
            'Grossular':       48.10436758915681,
            'Forsterite':      29.306222154198444,
        }

        pass1 = {sym: self.engine2.thermoPropertiesReaction(T, 0, sym).log_equilibrium_constant.val
                 for sym in expected}
        pass2 = {sym: self.engine2.thermoPropertiesReaction(T, 0, sym).log_equilibrium_constant.val
                 for sym in reversed(list(expected))}

        for sym, want in expected.items():
            assert pass1[sym] == pytest.approx(want, 1e-5, 1e-14)
            assert pass2[sym] == pytest.approx(want, 1e-5, 1e-14)

        values = list(pass1.values())
        assert len(set(values)) == len(values)

    def test_repeated_call_same_symbol_matches_first_call(self):
        # A cache hit must return exactly what the (memoized) miss returned -
        # this is memoize()/memoizeN() acting as a pure cache, independent of
        # the P write-back tested on the C++ side.
        first = self.engine.thermoPropertiesSubstance(873.15, 5000e5, "Quartz").gibbs_energy.val
        for _ in range(5):
            again = self.engine.thermoPropertiesSubstance(873.15, 5000e5, "Quartz").gibbs_energy.val
            assert again == first
