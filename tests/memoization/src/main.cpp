// Regression tests for the memoization helpers in ThermoFun/OptimizationUtils.h.
//
// These pin down the contract that ThermoEngine.cpp relies on for its four
// memoized property functions, whose signature is
//
//     std::function<Ret(double T, double P_, double& P, std::string symbol)>
//
// The third argument is an in-out parameter: callers pass P == 0 to mean "at
// the saturation pressure", and the solvent models write the computed Psat back
// through the reference (ThermoModelsSolvent.cpp:124, WaterHGK-JNgems.cpp:417).
//
// So the memoized wrapper has to get two separate things right:
//   1. the cache is keyed on the argument *values* at the moment of the call,
//      never on the caller's reference itself, and
//   2. a cache hit still has to deliver the post-call value of P, even though
//      the wrapped function does not run.

#include <cassert>
#include <cstdio>
#include <string>
#include <type_traits>

#include "ThermoFun/OptimizationUtils.h"

namespace {

// Mirrors ThermoPropertiesSubstanceFunction (ThermoEngine.cpp:49-50).
struct Props { double gibbs_energy = 0.0; };
using PropsFn = std::function<Props(double, double, double&, std::string)>;

// The saturation pressure the fake solvent model reports for a given T, in the
// same spirit as waterSaturatedPressureWagnerPruss(): only T determines it.
double psat(double T) { return 0.0061 * T; }

int calls = 0;

// Stands in for ThermoEngine's property lambdas: honours the P == 0 convention
// by writing the saturation pressure back through the reference.
Props compute(double T, double /*P_*/, double& P, std::string symbol)
{
    ++calls;
    if (P == 0.0)
        P = psat(T);
    Props p;
    p.gibbs_energy = T * 1000.0 + P + static_cast<double>(symbol.size());
    return p;
}

// Scribble over the stack region a just-returned frame occupied, so that a key
// holding a dangling reference cannot keep comparing equal by luck.
void clobberDeadFrame()
{
    volatile double scratch[128];
    for (int i = 0; i < 128; ++i) scratch[i] = i * 7.7;
    (void)scratch;
}

// ---------------------------------------------------------------------------
// 1. The cache key must not contain a reference.
//
// This is the original defect: with std::tuple<Args...> the key held a
// reference into the caller's stack frame, so entries dangled as soon as that
// frame died and std::map's ordering invariant was decided by freed memory.
// ---------------------------------------------------------------------------
template <typename Ret, typename... Args>
auto keyOf(std::function<Ret(Args...)>) -> std::tuple<std::decay_t<Args>...>;
using Key = decltype(keyOf(std::declval<PropsFn>()));

static_assert(!std::is_reference_v<std::tuple_element_t<2, Key>>,
              "the memoization cache key still holds a reference to the caller's P");
static_assert(std::is_same_v<Key, std::tuple<double, double, double, std::string>>,
              "the memoization cache key must be a tuple of plain values");

// ---------------------------------------------------------------------------
// 2. Entries stay valid once the calling frame is gone.
//
// Pre-fix this returned the last-inserted substance's properties for every
// symbol, because find() walked a tree ordered by dead stack slots.
// ---------------------------------------------------------------------------
template <typename Memoized>
void testSurvivesCallerFrame(Memoized memo, const char* which)
{
    calls = 0;

    // First call chain: P is a local of this block and dies with it.
    {
        double P = 1e5;
        memo(298.15, P, P, "H2O");
        memo(298.15, P, P, "Calcite");
    }
    assert(calls == 2);
    clobberDeadFrame();

    // Second call chain, a fresh lvalue holding the same value: both must hit.
    double P = 1e5;
    const Props h2o = memo(298.15, P, P, "H2O");
    const Props cal = memo(298.15, P, P, "Calcite");

    assert(calls == 2 && "cache missed: keys did not compare equal across frames");
    assert(h2o.gibbs_energy != cal.gibbs_energy &&
           "distinct substances came back with identical properties");
    assert(h2o.gibbs_energy == 298.15 * 1000.0 + 1e5 + 3.0);
    assert(cal.gibbs_energy == 298.15 * 1000.0 + 1e5 + 7.0);

    printf("  [ok] %s: entries survive the calling frame\n", which);
}

// ---------------------------------------------------------------------------
// 3. A cache hit must still write the in-out P back.
//
// The wrapped function does not run on a hit, so the wrapper itself has to
// replay the post-call value of P. Without that, a caller passing P == 0 gets
// the cached properties but is left holding 0 instead of Psat.
// ---------------------------------------------------------------------------
template <typename Memoized>
void testWritesBackOutArgument(Memoized memo, const char* which)
{
    calls = 0;
    const double T = 298.15;
    const double expected = psat(T);

    double first = 0.0;
    const Props a = memo(T, first, first, "H2O");
    assert(calls == 1);
    assert(first == expected && "the wrapped function must still see the reference");

    // Same call value (0.0), so this must be a hit - and must still yield Psat.
    double second = 0.0;
    const Props b = memo(T, second, second, "H2O");
    assert(calls == 1 && "keyed on the post-call value of P instead of the call value");
    assert(second == expected && "cache hit did not write the in-out P back");
    assert(a.gibbs_energy == b.gibbs_energy);

    // An argument taken by value must not be disturbed by the write-back.
    double byValue = 1e5;
    double byRef = 0.0;
    memo(T, byValue, byRef, "H2O");
    assert(byValue == 1e5 && "a by-value argument was overwritten");

    printf("  [ok] %s: cache hit writes the in-out P back\n", which);
}

// ---------------------------------------------------------------------------
// 4. memoizeN still evicts, and evicted entries recompute correctly.
// ---------------------------------------------------------------------------
void testEviction()
{
    calls = 0;
    auto memo = ThermoFun::memoizeN(PropsFn(compute), 2);

    double P = 1e5;
    memo(298.15, P, P, "A");    // fills
    memo(298.15, P, P, "BB");   // fills
    assert(calls == 2);

    memo(298.15, P, P, "CCC");  // evicts the least recently used entry
    assert(calls == 3);

    // Whatever survived must still be correct rather than a stale neighbour.
    const Props ccc = memo(298.15, P, P, "CCC");
    assert(ccc.gibbs_energy == 298.15 * 1000.0 + 1e5 + 3.0);

    printf("  [ok] memoizeN: eviction keeps surviving entries correct\n");
}

} // namespace

int main()
{
    printf("memoization regression tests\n");

    testSurvivesCallerFrame(ThermoFun::memoize(PropsFn(compute)), "memoize");
    testSurvivesCallerFrame(ThermoFun::memoizeN(PropsFn(compute), size_t(1e6)), "memoizeN");

    testWritesBackOutArgument(ThermoFun::memoize(PropsFn(compute)), "memoize");
    testWritesBackOutArgument(ThermoFun::memoizeN(PropsFn(compute), size_t(1e6)), "memoizeN");

    testEviction();

    printf("all memoization tests passed\n");
    return 0;
}
