import std.typecons;
import std.exception, std.math;
import std.conv : to;
import std.algorithm, std.array, std.range, std.string;
import std.container.binaryheap;
import std.stdio;
import std.getopt;

import std.experimental.logger;
alias Data = Tuple!(real, "p", real, "sigmaStar");

void main(string[] args)
{
    auto pColumn = size_t.max, sigmaStarColumn = size_t.max, rhoStarColumn = size_t.max;
    auto gor = args.getopt("p", &pColumn, "sigma|s", &sigmaStarColumn, "rho|r", &rhoStarColumn);
    if (gor.helpWanted)
    {
        "ema".defaultGetoptPrinter(gor.options);
        return;
    }
    string fileName = args[1];

    Data[] data;
    bool notFirst;
    foreach (line; File(fileName).byLine)
    {
        auto buf = line.chomp.split("\t");
        if (!notFirst)
        {
            if (sigmaStarColumn != size_t.max)
                "%s:%s".infof(buf[pColumn], buf[sigmaStarColumn]);
            if (rhoStarColumn != size_t.max)
                "%s:%s".infof(buf[pColumn], buf[rhoStarColumn]);
            notFirst = true;
            continue;
        }
        if (sigmaStarColumn != size_t.max)
            data ~= Data(buf[pColumn].to!real, buf[sigmaStarColumn].to!real);
        if (rhoStarColumn != size_t.max)
            data ~= Data(buf[pColumn].to!real, 1/buf[rhoStarColumn].to!real);
    }

    immutable
        i0 = data.countUntil!(a => a.p == 0),
        i1 = data.countUntil!(a => a.p == 1),
        sigma0 = data[i0].sigmaStar,
        sigma1 = data[i1].sigmaStar;
    data = data.remove(i0.max(i1)).remove(i0.min(i1));
    auto results = [fitPredict(sigma0, sigma1, data)];
    foreach (i; 0..data.length)
        results ~= fitPredict(sigma0, sigma1, data[0..i] ~ data[i+1..$]);
    "p".write;
    foreach (result; results)
    {
        "\t(%(%e,%))".writef(result.params[]);
    }
    writeln;
    foreach (i; 0..99)
    {
        "%f".writef(results[0].result[i].p);
        foreach (result; results)
        {
            "\t%e".writef(result.result[i].sigmaStar);
        }
        writeln;
    }
}

auto fitPredict(in real sigma0, in real sigma1, Data[] data)
{
    Data[] result;
    real[3] lowerBound = [0.0001, 1, -9],
            upperBound = [4, 1000, -7];
    auto searcher = Searcher!3(lowerBound, upperBound, 16, 32, 8);
    auto model = new EMA3P(sigma1);
    auto seachState = searcher.search(model, data);
    auto params = center(seachState.lowerBound, seachState.upperBound);
    foreach (i; 1..100)
    {
        immutable p = i / 100.0L;
        try
            result ~= Data(p, model.sigmaStarTheoretical(params, p));
        catch (Throwable t)
        {
            "%s".warningf(t);
        }
    }
    return tuple!("params", "result")(params, result);
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

class EMA3P : Model!(3, Data)
{
    this (in real sigma1=0)
    {
        this.sigma1 = sigma1;
    }
    real sigmaStarTheoretical(real[3] params, real p)
    {
        immutable
            sigma0 = 10 ^^ params[2],
            sigma2 = (10 ^^ params[0]) * sigma0,
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
    real error(real[3] params, Tuple!(real, "p", real, "sigmaStar") d)
    {
        return sigmaStarTheoretical(params, d.p) - d.sigmaStar;
    }
    immutable real sigma1;
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
