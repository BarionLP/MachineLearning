using System.Diagnostics;
using System.Runtime.CompilerServices;
using MachineLearning.Model.Activation;

namespace MachineLearning.Model.Initialization;

public enum FanMode { FanIn, FanOut, FanAvg }
public static class TensorInitializer
{
    public static void Uniform(this Vector v, Weight low, Weight high, Random random)
    {
        v.MapToSelf(_ => low + (high - low) * random.NextSingle());
    }
    public static void Uniform(this Matrix m, Weight low, Weight high, Random random)
    {
        m.Storage.Uniform(low, high, random);
    }
    public static void Uniform(this Tensor t, Weight low, Weight high, Random random)
    {
        t.Storage.Uniform(low, high, random);
    }


    public static void Normal(this Vector v, Weight mean, Weight std, Random random)
    {
        v.MapToSelf(_ => mean + std * NextGaussian(random));
    }
    public static void Normal(this Matrix m, Weight mean, Weight std, Random random)
    {
        m.Storage.Normal(mean, std, random);
    }
    public static void Normal(this Tensor t, Weight mean, Weight std, Random random)
    {
        t.Storage.Normal(mean, std, random);
    }

    public static void XavierUniform(this Matrix w, Random random)
    {
        var (fanIn, fanOut) = Fans(w);
        var bound = Weight.Sqrt(6f / (fanIn + fanOut));
        w.Uniform(-bound, bound, random);
    }
    public static void XavierNormal(this Matrix w, Random random)
    {
        var (fanIn, fanOut) = Fans(w);
        var std = Weight.Sqrt(2f / (fanIn + fanOut));
        w.Normal(0, std, random);
    }

    public static void KaimingUniform(this Matrix w, IActivationFunction nl, Random random, FanMode mode = FanMode.FanIn, Weight a = 0)
    {
        var fan = Fan(w, mode);
        var gain = Gain(nl, a);
        var bound = gain * Weight.Sqrt(6f / fan);
        w.Uniform(-bound, bound, random);
    }
    public static void KaimingNormal(this Matrix w, IActivationFunction nl, Random random, FanMode mode = FanMode.FanIn, Weight a = 0)
    {
        var fan = Fan(w, mode);
        var gain = Gain(nl, a);
        var std = gain * Weight.Sqrt(2f / fan);
        w.Normal(0, std, random);
    }

    public static void LeCunUniform(this Matrix w, Random random, FanMode mode = FanMode.FanIn)
    {
        var fan = Fan(w, mode);
        var bound = Weight.Sqrt(3f / fan);
        w.Uniform(-bound, bound, random);
    }
    public static void LeCunNormal(this Matrix w, Random random, FanMode mode = FanMode.FanIn)
    {
        var fan = Fan(w, mode);
        var std = Weight.Sqrt(1f / fan);
        w.Normal(0, std, random);
    }

    public static void XavierUniform(this Tensor t, int fanInAxis, int fanOutAxis, Random random)
    {
        var (fi, fo) = FansForTensor(t, fanInAxis, fanOutAxis);
        var bound = Weight.Sqrt(6f / (fi + fo));
        t.Uniform(-bound, bound, random);
    }
    public static void XavierNormal(this Tensor t, int fanInAxis, int fanOutAxis, Random random)
    {
        var (fi, fo) = FansForTensor(t, fanInAxis, fanOutAxis);
        var std = Weight.Sqrt(2f / (fi + fo));
        t.Normal(0, std, random);
    }
    public static void KaimingUniform(this Tensor t, int fanInAxis, IActivationFunction nl, Random random, FanMode mode = FanMode.FanIn, Weight a = 0)
    {
        var (fi, fo) = FansForTensor(t, fanInAxis, mode is FanMode.FanOut ? fanInAxis : fanInAxis);
        int fan = mode switch
        {
            FanMode.FanIn => fi,
            FanMode.FanOut => fo,
            FanMode.FanAvg => (fi + fo) / 2,
            _ => throw new UnreachableException()
        };
        var bound = Gain(nl, a) * Weight.Sqrt(6 / fan);
        t.Uniform(-bound, bound, random);
    }
    public static void KaimingNormal(this Tensor t, int fanInAxis, IActivationFunction nl, Random random, FanMode mode = FanMode.FanIn, Weight a = 0)
    {
        var (fi, fo) = FansForTensor(t, fanInAxis, mode is FanMode.FanOut ? fanInAxis : fanInAxis);
        int fan = mode switch
        {
            FanMode.FanIn => fi,
            FanMode.FanOut => fo,
            FanMode.FanAvg => (fi + fo) / 2,
            _ => throw new UnreachableException()
        };
        var std = Gain(nl, a) * Weight.Sqrt(2 / fan);
        t.Normal(0, std, random);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static (int fanIn, int fanOut) Fans(Matrix m) => (m.ColumnCount, m.RowCount); // output x input → fan_in = inputs, fan_out = outputs

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static (int fanIn, int fanOut) FansForTensor(Tensor t, int fanInAxis, int fanOutAxis)
    {
        ReadOnlySpan<int> dims = [t.RowCount, t.ColumnCount, t.LayerCount];
        return (dims[fanInAxis], dims[fanOutAxis]);
    }

    static int Fan(Matrix w, FanMode mode)
    {
        var (fi, fo) = Fans(w);
        return mode switch
        {
            FanMode.FanIn => fi,
            FanMode.FanOut => fo,
            FanMode.FanAvg => (fi + fo) / 2,
            _ => fi
        };
    }

    static Weight Gain(IActivationFunction n, Weight a) => n switch
    {
        SigmoidActivation => 1,
        TanhActivation => 5 / 3,
        ReLUActivation => Weight.Sqrt(2f),
        LeakyReLUActivation => Weight.Sqrt(2 / (1 + a * a)),
        // Nonlinearity.GELU => Weight.Sqrt(2.0),   // common approx
        // Nonlinearity.Swish => Weight.Sqrt(2.0),  // reasonable default
        _ => throw new NotImplementedException(),
    };

    static Weight NextGaussian(Random r)
    {
        var u1 = 1 - r.NextSingle();
        var u2 = 1 - r.NextSingle();
        return Weight.Sqrt(-2 * Weight.Log(u1)) * Weight.Cos(2 * MathF.PI * u2);
    }
}
