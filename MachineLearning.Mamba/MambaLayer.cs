using System.Collections.Immutable;
using System.Diagnostics;
using MachineLearning.Model;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Mamba;

public sealed class MambaLayer(int sequenceLength, int stateDimensions) : ILayer
{
    public int SequenceLength /*T*/ { get; } = sequenceLength;
    public int StateDimensions /*N*/ { get; } = stateDimensions;
    public Vector Alpha { get; } = Vector.Create(sequenceLength);
    public Matrix B { get; } = Matrix.Create(sequenceLength, stateDimensions);
    public Matrix C { get; } = Matrix.Create(sequenceLength, stateDimensions);

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count == SequenceLength);

        input.CopyTo(snapshot.Input);
        snapshot.H.ResetZero();

        for (int t = 0; t < SequenceLength; t++)
        {
            // h = alpha_t * h + B_t * x_t
            var h = snapshot.H.RowRef(t);
            if (t > 0)
            {
                snapshot.H.RowRef(t - 1).MultiplyTo(Alpha[t], h);
            }

            h.AddToSelf(B.RowRef(t).Multiply(input[t])); // add B_t * x_t

            // y_t = C_t^T * h
            snapshot.Output[t] = C.RowRef(t).Dot(h);
        }

        return snapshot.Output;
    }

    public void BackwardPass(Snapshot st, Vector dOutput)
    {
        Debug.Assert(dOutput.Count == SequenceLength);

        // 1. Clear all parameter gradient buffers
        st.dAlpha.ResetZero();
        st.dX.ResetZero();
        st.dB.ResetZero();
        st.dC.ResetZero();
        st.dH.ResetZero();

        for (int t = SequenceLength - 1; t >= 0; t--)
        {
            // let dY = derivative of L wrt outputY[t]
            float dY = dOutput[t];

            // (a) outputY[t] = dot(C[t], h[t])
            // => dC[t] += h[t] * dY
            // => dH[t] += C[t] * dY
            st.dC.RowRef(t).AddToSelf(st.H.RowRef(t).Multiply(dY));
            st.dH.RowRef(t).AddToSelf(C.RowRef(t).Multiply(dY));

            // (b) h[t] = alpha[t] * h[t-1] + B[t] * inputX[t]
            // => partial wrt alpha[t] = (h[t-1] dot dH[t])
            // => partial wrt h[t-1]  += alpha[t] * dH[t]
            // => partial wrt B[t]    += dH[t] * inputX[t]
            // => partial wrt inputX[t] = (B[t] dot dH[t])

            // derivative w.r.t alpha[t]
            // h[t-1] = (t>0) ? st.h[t-1] : zero
            var hPrev = (t == 0)
                           ? Vector.Create(StateDimensions)
                           : st.H.RowRef(t - 1);

            st.dAlpha[t] += hPrev.Dot(st.dH.RowRef(t));  // dAlpha

            // derivative w.r.t h[t-1]
            // if t>0, add alpha[t]*dH[t] to dH[t-1]
            if (t > 0)
            {
                st.H.RowRef(t - 1).AddToSelf(st.dH.RowRef(t).Multiply(Alpha[t]));
            }

            // derivative w.r.t. B[t] and Input[t]
            // dB[t][i] = Input[t] * dH[t][i]
            st.dB.RowRef(t).AddToSelf(st.dH.RowRef(t).Multiply(st.Input[t]));

            // accumulate dot for dX[t]
            st.dX[t] += B.RowRef(t).Dot(st.dH.RowRef(t));  // partial w.r.t. Input[t]
        }
    }

    public long ParameterCount { get; }
    public ILayerSnapshot CreateSnapshot() => new Snapshot(SequenceLength, StateDimensions);

    public sealed class Snapshot(int T, int N) : ILayerSnapshot
    {
        public Vector Input { get; } = Vector.Create(T);

        // Hidden states for forward pass
        public Matrix H { get; } = Matrix.Create(T, N);
        public Vector Output { get; } = Vector.Create(T);

        // For convenience, store partial derivatives
        //   dAlpha[t], dB[t], dC[t], dX[t]
        public Vector dAlpha { get; } = Vector.Create(T);
        public Matrix dB { get; } = Matrix.Create(T, N);
        public Matrix dC { get; } = Matrix.Create(T, N);
        public Vector dX { get; } = Vector.Create(T);

        // Hidden-state partials for backprop
        public Matrix dH { get; } = Matrix.Create(T, N);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<MambaLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(MambaLayer layer)
        {
            var sqrtInputNodeCount = MathF.Sqrt(layer.SequenceLength);

            layer.Alpha.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0f, 1f) / sqrtInputNodeCount);
            layer.B.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0f, 0.1f) / sqrtInputNodeCount);
            layer.C.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0f, 0.1f) / sqrtInputNodeCount);
        }
    }

}


public sealed class Mamba2Model(int layerCount, int contextSize, int dims) : IModel<Vector, MambaLayer.Snapshot>
{
    public ImmutableArray<MambaLayer> Layers { get; } = [.. Enumerable.Range(0, layerCount).Select(_ => new MambaLayer(contextSize, dims))];

    public Vector Process(Vector input)
    {

        throw new NotImplementedException();
    }

    public Vector Process(Vector input, ImmutableArray<MambaLayer.Snapshot> snapshots)
    {

        return Layers.Zip(snapshots).Aggregate(input, (v, l) => l.First.Forward(v, l.Second));
    }

    public long ParameterCount => throw new NotImplementedException();
}

public sealed class Mamba2LayerAdam : ILayerOptimizer<MambaLayer, MambaLayer.Snapshot>
{
    public MambaLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public AdamOptimizer Optimizer { get; }

    public Vector GradientAlpha { get; }
    public Matrix GradientB { get; }
    public Matrix GradientC { get; }

    // formula symbol M 
    // exponentially decaying average of past gradients. It is akin to the mean of the gradients.
    public readonly Vector FirstMomentAlpha;
    public readonly Matrix FirstMomentB;
    public readonly Matrix FirstMomentC;

    // formula symbol V
    // exponentially decaying average of the squared gradients. It is akin to the uncentered variance of the gradients.
    public readonly Vector SecondMomentAlpha;
    public readonly Matrix SecondMomentB;
    public readonly Matrix SecondMomentC;


    public Mamba2LayerAdam(AdamOptimizer optimizer, MambaLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientAlpha = Vector.Create(Layer.SequenceLength);
        GradientB = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
        GradientC = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);

        FirstMomentAlpha = Vector.Create(Layer.SequenceLength);
        FirstMomentB = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
        FirstMomentC = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);

        SecondMomentAlpha = Vector.Create(Layer.SequenceLength);
        SecondMomentB = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
        SecondMomentC = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
    }

    private readonly Lock _lock = new();
    public void Update(Vector nodeValues, MambaLayer.Snapshot snapshot)
    {
        Layer.BackwardPass(snapshot, nodeValues);
        // Compute the gradient for weights
        //VectorHelper.MultiplyToMatrixTo(nodeValues, snapshot.LastRawInput, snapshot.WeightGradients); // GradientCostWeights.AddInPlaceMultiplied ?

        NumericsDebug.AssertValidNumbers(nodeValues);
        NumericsDebug.AssertValidNumbers(snapshot.dAlpha);
        NumericsDebug.AssertValidNumbers(snapshot.dB);
        NumericsDebug.AssertValidNumbers(snapshot.dC);

        lock (_lock)
        {
            GradientAlpha.AddToSelf(snapshot.dAlpha);
            GradientB.AddToSelf(snapshot.dB);
            GradientC.AddToSelf(snapshot.dC);
        }
    }

    // update child methods
    public void Apply(int dataCounter)
    {
        // do i need gradient clipping?
        var averagedLearningRate = Optimizer.LearningRate / Weight.Sqrt(dataCounter);

        // Update biases
        (FirstMomentAlpha, GradientAlpha).MapToFirst(FirstMomentEstimate);
        (SecondMomentAlpha, GradientAlpha).MapToFirst(SecondMomentEstimate);
        Layer.Alpha.SubtractToSelf((FirstMomentAlpha, SecondMomentAlpha).Map(WeightReduction));

        (FirstMomentB, GradientB).MapToFirst(FirstMomentEstimate);
        (SecondMomentB, GradientB).MapToFirst(SecondMomentEstimate);
        Layer.B.SubtractToSelf((FirstMomentB, SecondMomentB).Map(WeightReduction));

        (FirstMomentC, GradientC).MapToFirst(FirstMomentEstimate);
        (SecondMomentC, GradientC).MapToFirst(SecondMomentEstimate);
        Layer.C.SubtractToSelf((FirstMomentC, SecondMomentC).Map(WeightReduction));

        Weight WeightReduction(Weight firstMoment, Weight secondMoment)
        {
            var mHat = firstMoment / (1 - Weight.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - Weight.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
            return averagedLearningRate * mHat / (Weight.Sqrt(vHat) + Optimizer.Epsilon);
        }
        Weight FirstMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

        Weight SecondMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;
    }

    public void GradientCostReset()
    {
        GradientAlpha.ResetZero();
        GradientB.ResetZero();
        GradientC.ResetZero();
    }

    public void FullReset()
    {
        GradientCostReset();

        FirstMomentAlpha.ResetZero();
        SecondMomentAlpha.ResetZero();

        FirstMomentB.ResetZero();
        SecondMomentB.ResetZero();
        FirstMomentC.ResetZero();
        SecondMomentC.ResetZero();
    }
}