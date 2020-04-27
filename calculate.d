import std.typecons : Tuple;
import std.exception, std.math;
import std.conv : to;
import std.algorithm, std.array, std.range, std.string;
import std.container.binaryheap;

import std.experimental.logger;

void main(string[] args)
{
    import std.stdio;

    string fileName = args[1];
    immutable pColumn = args[2].to!size_t, sigmaStarColumn = args[3].to!size_t;
    alias Data = Tuple!(real, "p", real, "sigmaStar");
    Data[] data;
    bool notFirst;
    foreach (line; File(fileName).byLine)
    {
        auto buf = line.chomp.split("\t");
        if (!notFirst)
        {
            "%s:%s".infof(buf[pColumn], buf[sigmaStarColumn]);
            notFirst = true;
            continue;
        }
        data ~= Data(buf[pColumn].to!real, buf[sigmaStarColumn].to!real);
    }

    immutable real sigma0 = data[0].sigmaStar, sigma1 = data[$-1].sigmaStar;
    data = data[1..$-1];
    real[2] lowerBound = [sigma0*1.0001, 1],
            upperBound = [1e-2, 1000];
    auto searcher = Searcher!2(lowerBound, upperBound, 16, 16, 4);
    auto model = new EMA3(sigma0, sigma1);
    auto seachState = searcher.search(model, data);
    auto params = center(seachState.lowerBound, seachState.upperBound);
    "p\tsigmastar(sigma2=%e,m=%e)".writefln(params[0], params[1]);
    foreach (i; 1..100)
    {
        immutable p = i / 100.0L;
        try
            "%f\t%e".writefln(p, model.sigmaStarTheoretical(params, p));
        catch (Throwable t)
        {
            "%s".warningf(t);
        }
    }
}

struct Searcher(size_t numParams)
{
    real[numParams] lowerBound, upperBound;
    size_t gridSize, beamSize, depth;
    alias SearchState = Tuple!(
            real, "meanSquareError",
            real[numParams], "lowerBound",
            real[numParams], "upperBound"
        );

    auto search(Model, Data)(Model model, Data data)
    {
        alias BH = BinaryHeap!(SearchState[], ((a, b) => a.meanSquareError < b.meanSquareError));
        auto h = BH([SearchState(
                    model.meanSquareError(center(lowerBound, upperBound), data),
                    lowerBound, upperBound)], 1);
        foreach (d; 0..depth)
        {
            "depth: %d".infof(d);
            auto hh = BH(new SearchState[beamSize], 0);
            foreach (searchState; h)
            {
                "state: %e (%s, %s)".infof(searchState.meanSquareError, searchState.lowerBound, searchState.upperBound);
                foreach (newLower, newUpper; Grid!numParams(searchState.lowerBound, searchState.upperBound, gridSize))
                {
                    hh.conditionalInsert(
                            SearchState(model.meanSquareError(center(newLower, newUpper), data),
                                newLower, newUpper)
                        );
                }
            }
            h = hh;
        }
        return h.array[$-1];
    }
}

real[m] center(size_t m)(real[m] lower, real[m] upper)
{
    real[m] ret;
    foreach (i; 0..m)
        ret[i] = (lower[i] + upper[i]) / 2;
    return ret;
}

struct Grid(size_t m)
{
    this (real[m] lower, real[m] upper, size_t size)
    {
        this.lower = lower;
        this.upper = upper;
        this.size = size;
        foreach (i; 0..m)
            foreach (j; 0..size)
            {
                _lowers[i] ~= (lower[i] * (size-j) + upper[i] * j) / size;
                _uppers[i] ~= (lower[i] * (size-j-1) + upper[i] * (j+1)) / size;
            }
    }
    int opApply(scope int delegate(ref real[m], ref real[m]) dg)
    {
        real[m] _lower, _upper;
        size_t counterMax = 1;
        foreach (i; 0..m)
            counterMax *= size;
        foreach (counter; 0..counterMax)
        {
            auto c = counter;
            foreach (i; 0..m)
            {
                _lower[i] = _lowers[i][c % size];
                _upper[i] = _uppers[i][c % size];
                c /= size;
            }
            if (auto result = dg(_lower, _upper))
                return result;
        }
        return 0;
    }
    immutable real[m] lower, upper;
    immutable size_t size;
    private real[][m] _lowers, _uppers;
}

interface Model(size_t numParams, Data)
{
    final real meanSquareError(real[numParams] params, Data[] data)
    {
        real sum = 0;
        foreach (d; data)
            sum += error(params, d) ^^ 2.0L;
        return sum / data.length;
    }
    real error(real[numParams] params, Data d);
}

class EMA3 : Model!(2, Tuple!(real, "p", real, "sigmaStar"))
{
    this (in real sigma0, in real sigma1)
    {
        this.sigma0 = sigma0;
        this.sigma1 = sigma1;
    }
    real sigmaStarTheoretical(real[2] params, real p)
    {
        immutable
            sigma2 = params[0],
            m = params[1];
        immutable
            p0 = (1 - p) ^^ m,
            p1 = p,
            p2 = 1 - (p0 + p1),
            q = p2 / p1;
        immutable
            sigmar = (1 - (1+q) ^^ (-2.0L/3)) * sigma2 +
                     (1+q) ^^ (-1.0L/3) / (
                        (1 / sigma1) + ((1+q) ^^ (1.0L/3) - 1) / sigma2);
        return positiveSolution(
                -2,
                p0 * 3 * (sigma0 - sigmar) + (2*sigmar - sigma0),
                sigma0 * sigmar
            );
    }
    /// sigma2, m
    real error(real[2] params, Tuple!(real, "p", real, "sigmaStar") d)
    {
        return sigmaStarTheoretical(params, d.p) - d.sigmaStar;
    }
    immutable real sigma0, sigma1;
}

/// Solve equation axx+bx+c=0; 0<x
real positiveSolution(in real a, in real b, in real c)
out (result)
{
    assert (0 < result, "2de(%f, %e, %e) = %e".format(a, b, c, result));
}
do
{
    if (a < 0)
        return positiveSolution(-a, -b, -c);
    //"2de(%f, %e, %e)".infof(a, b, c);
    (0 < a).enforce("degenerate equation");
    assert (0 <= b*b);
    (0 < -4 * a * c).enforce;
    immutable d = b * b - 4 * a * c;
    (0 <= d).enforce("imaginary solution");
    (-b - d.sqrt <= 0).enforce("both solutions are positive");
    return (-b + d.sqrt) / (2 * a);
}
