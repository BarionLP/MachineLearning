using MachineLearning.Model.Attributes;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Serialization;
using MachineLearning.Training.Attributes;

namespace MachineLearning.Mamba;

[GeneratedLayer, LayerSerializer("vmam2", 3), GenerateOptimizers]
public sealed partial class Mamba2VectorLayer : ILayer<Matrix, Mamba2VectorLayer.Snapshot>
{
    [Parameter] public int MaxSequenceLength /*T*/ { get; }
    public int StateDimensions /*N*/ => W_A.ColumnCount;
    public int EmbeddingDimensions /*E*/ => W_A.RowCount;
    [Weights] public Matrix W_A { get; }
    [Weights] public Matrix W_B { get; }
    [Weights] public Matrix W_C { get; }
    [Weights] public Matrix W_X { get; }
    [Weights] public Matrix W_O { get; }

    public Mamba2VectorLayer(int sequenceLength, int stateDimensions, int embeddingDimensions)
        : this(sequenceLength, Matrix.Create(embeddingDimensions, stateDimensions), Matrix.Create(embeddingDimensions, stateDimensions), Matrix.Create(embeddingDimensions, stateDimensions), Matrix.Create(embeddingDimensions, stateDimensions), Matrix.Create(embeddingDimensions, stateDimensions)) { }

    public Matrix Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.RowCount <= MaxSequenceLength);
        Debug.Assert(input.ColumnCount == EmbeddingDimensions);

        snapshot.Input = input;
        snapshot.Memory.ResetZero();

        for (int t = 0; t < snapshot.SequenceLength; t++)
        {
            var x_t = input.RowRef(t);

            var A_hat = snapshot.Alpha.RowRef(t);
            var B_hat = snapshot.B_Hat.RowRef(t);
            var C_hat = snapshot.C_Hat.RowRef(t);
            var X_t = snapshot.X.RowRef(t);

            W_A.MultiplyTransposedTo(x_t, A_hat);
            W_B.MultiplyTransposedTo(x_t, B_hat);
            W_C.MultiplyTransposedTo(x_t, C_hat);
            W_X.MultiplyTransposedTo(x_t, X_t);

            var k = snapshot.K.RowRef(t);
            var v = snapshot.V.RowRef(t);

            A_hat.MapToSelf(value => 1 / (1 + Weight.Exp(value)));
            B_hat.SwishTo(k);
            C_hat.SwishTo(v);

            // h = alpha_t * h + B * x_t
            var h = snapshot.Memory.RowRef(t);
            if (t > 0)
            {
                Debug.Assert(h.Sum() == 0);
                snapshot.Memory.RowRef(t - 1).PointwiseMultiplyTo(A_hat, h);
            }


            k.PointwiseMultiplyAddTo(X_t, h); // h += B * x_t

            var gated = snapshot.Gated.RowRef(t);
            v.PointwiseMultiplyTo(h, gated);
            W_O.MultiplyTo(gated, snapshot.Output.RowRef(t));
        }

        NumericsDebug.AssertValidNumbers(snapshot.Output);
        return snapshot.Output.Rows(..snapshot.SequenceLength);
    }

    private Vector Zero
    {
        get
        {
            field ??= Vector.Create(StateDimensions);
            return field;
        }
    }

    public Matrix Backward(Matrix outputGradient, Snapshot snapshot, Gradients gradients)
    {
        Debug.Assert(outputGradient.RowCount <= MaxSequenceLength);
        Debug.Assert(outputGradient.ColumnCount == EmbeddingDimensions);

        snapshot.GradientInput.ResetZero();
        snapshot.GradientMemory.ResetZero();

        for (int t = snapshot.SequenceLength - 1; t >= 0; t--)
        {
            var outputGradient_t = outputGradient.RowRef(t);
            var memoryGradient_t = snapshot.GradientMemory.RowRef(t);

            var gated = snapshot.Gated.RowRef(t);
            var dGated = Vector.Create(gated.Count);

            W_O.MultiplyTransposedTo(outputGradient_t, dGated);                  // dGated = W_O * dY
            VectorHelper.MultiplyToMatrixAddTo(outputGradient_t, gated, gradients.W_O);   // dW_O += dY ⊗ gated

            snapshot.V.RowRef(t).PointwiseMultiplyAddTo(dGated, memoryGradient_t);        // dH += dY ⊙ v
            var dV = snapshot.Memory.RowRef(t).PointwiseMultiply(dGated);                 // dV = dY ⊙ h

            var hPrev = (t == 0) ? Zero : snapshot.Memory.RowRef(t - 1);
            var dAlpha = hPrev.PointwiseMultiply(memoryGradient_t);        // dα  = dH ⊙ h_{t-1}
            if (t > 0)
            {
                memoryGradient_t.PointwiseMultiplyAddTo(snapshot.Alpha.RowRef(t), snapshot.GradientMemory.RowRef(t - 1)); // dH_{t-1} += dH ⊙ α
            }

            var dK = snapshot.X.RowRef(t).PointwiseMultiply(memoryGradient_t);                // dK = dH ⊙ X
            var dX = snapshot.K.RowRef(t).PointwiseMultiply(memoryGradient_t);                // dX = dH ⊙ k

            SigmoidPrimeInPlace(snapshot.Alpha.RowSpan(t), dAlpha.AsSpan());                 // dÂ
            dAlpha.MultiplyToSelf(-1f);
            SwishPrimeInPlace(snapshot.B_Hat.RowSpan(t), dK.AsSpan());                        // dB̂
            SwishPrimeInPlace(snapshot.C_Hat.RowSpan(t), dV.AsSpan());                        // dĈ

            VectorHelper.MultiplyToMatrixAddTo(snapshot.Input.RowRef(t), dAlpha, gradients.W_A);
            VectorHelper.MultiplyToMatrixAddTo(snapshot.Input.RowRef(t), dK, gradients.W_B);
            VectorHelper.MultiplyToMatrixAddTo(snapshot.Input.RowRef(t), dV, gradients.W_C);
            VectorHelper.MultiplyToMatrixAddTo(snapshot.Input.RowRef(t), dX, gradients.W_X);
            var inputGradient_t = snapshot.GradientInput.RowRef(t);
            W_A.MultiplyAddTo(dAlpha, inputGradient_t);
            W_B.MultiplyAddTo(dK, inputGradient_t);
            W_C.MultiplyAddTo(dV, inputGradient_t);
            W_X.MultiplyAddTo(dX, inputGradient_t);
        }

        return snapshot.GradientInput.Rows(..snapshot.SequenceLength);
    }

    static void SigmoidPrimeInPlace(ReadOnlySpan<float> s, Span<float> grad)
    {
        for (int i = 0; i < s.Length; i++) grad[i] *= s[i] * (1 - s[i]);
    }
    static void SwishPrimeInPlace(ReadOnlySpan<float> z, Span<float> grad)
    {
        for (int i = 0; i < z.Length; i++)
        {
            var σ = 1f / (1f + MathF.Exp(-z[i]));
            grad[i] *= σ * (1 + z[i] * (1 - σ));
        }
    }


    public partial class Snapshot
    {
        public int SequenceLength => Input.RowCount;
        public Matrix Input { get; set; } = null!;
        public Matrix GradientInput { get; } = Matrix.Create(layer.MaxSequenceLength, layer.EmbeddingDimensions);
        public Matrix Output { get; } = Matrix.Create(layer.MaxSequenceLength, layer.EmbeddingDimensions);

        public Matrix Memory /*H*/ { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions); // one row per timestep
        public Matrix GradientMemory { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);

        public Matrix Alpha { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);   // post-sigmoid α
        public Matrix B_Hat { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);   // pre‑swish B̂
        public Matrix C_Hat { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);   // pre‑swish Ĉ
        public Matrix V { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);
        public Matrix K { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);
        public Matrix X { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);
        public Matrix Gated { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<Mamba2VectorLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(Mamba2VectorLayer layer)
        {
            var scale = Weight.Sqrt(36 / ((Weight)layer.StateDimensions + layer.EmbeddingDimensions));

            layer.W_B.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0f, scale));
            layer.W_C.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0f, scale));
            layer.W_X.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0f, scale));
            layer.W_O.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0f, scale));

            float shift = -2.2f; // so that α = σ(-Â) ≈ 0.9 by default
            layer.W_A.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, shift, 0.5f * scale));
        }
    }

}
