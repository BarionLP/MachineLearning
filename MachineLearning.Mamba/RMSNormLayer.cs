using System.Numerics.Tensors;
using MachineLearning.Model.Attributes;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Serialization;
using MachineLearning.Training.Attributes;

namespace MachineLearning.Mamba;

[GeneratedLayer, LayerSerializer("rms", 1), GenerateOptimizers]
public sealed partial class RMSNormLayer : ILayer<Matrix, RMSNormLayer.Snapshot>
{
    public int EmbeddingDimensions => Gamma.Count;
    [Parameter] public int SequenceLength { get; }
    [Weights] public Vector Gamma { get; }

    public RMSNormLayer(int sequenceLength, int embeddingDimensions)
        : this(sequenceLength, Vector.Create(embeddingDimensions)) { }


    public Matrix Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.ColumnCount == EmbeddingDimensions);

        input.CopyTo(snapshot.Input);

        foreach (var t in ..input.RowCount)
        {
            var input_t = input.RowSpan(t);
            var output_t = snapshot.Output.RowSpan(t);
            RMSNorm.Normalize(input_t, output_t, Gamma.AsSpan());
        }

        return snapshot.Output;
    }

    public Matrix Backward(Matrix outputGradient, Snapshot snapshot, Gradients gradients)
    {
        Debug.Assert(outputGradient.ColumnCount == EmbeddingDimensions);

        gradients.Gamma.ResetZero();
        Span<Weight> gradientGamma = stackalloc Weight[gradients.Gamma.Count];

        foreach (var t in ..outputGradient.RowCount)
        {
            var input_t = snapshot.Input.RowSpan(t);
            var dinput_t = snapshot.GradientInput.RowSpan(t);
            var doutput_t = outputGradient.RowSpan(t);
            RMSNorm.Backward(input_t, doutput_t, Gamma.AsSpan(), dinput_t, gradientGamma);
            TensorPrimitives.Add(gradients.Gamma.AsSpan(), gradientGamma, gradients.Gamma.AsSpan());
        }

        return snapshot.GradientInput;
    }

    partial class Snapshot
    {
        public Matrix Input { get; } = Matrix.Create(layer.SequenceLength, layer.EmbeddingDimensions);
        public Matrix Output { get; } = Matrix.Create(layer.SequenceLength, layer.EmbeddingDimensions);
        public Matrix GradientInput { get; } = Matrix.Create(layer.SequenceLength, layer.EmbeddingDimensions);
        public Vector GetInputGradient() => GradientInput.Storage;
    }

    public sealed class Initializer : IInitializer<RMSNormLayer>
    {
        public void Initialize(RMSNormLayer layer)
        {
            layer.Gamma.Fill(0.1f);
        }
    }
}


public static class RMSNorm
{
    public static void Normalize(
        ReadOnlySpan<Weight> input,
        Span<Weight> output,
        ReadOnlySpan<Weight> gamma,
        Weight eps = 1e-6f)
    {
        int n = input.Length;

        var sumSq = TensorPrimitives.SumOfSquares(input);
        // Compute inverse root mean square (1/√(meanSquare + eps))
        var invRms = 1.0f / Weight.Sqrt((sumSq / n) + eps);

        TensorPrimitives.Multiply(input, invRms, output);
        TensorPrimitives.Multiply(output, gamma, output);
    }

    public static void Backward(
        ReadOnlySpan<Weight> input,
        ReadOnlySpan<Weight> outputGradient,  // dL/dy
        ReadOnlySpan<Weight> gamma,
        Span<Weight> inputGradient,           // dL/dx (output)
        Span<Weight> gammaGradient,           // dL/dgamma (output)
        Weight eps = 1e-6f)
    {
        int n = input.Length;

        // Recompute RMS norm (or retrieve from forward pass if available)
        float sumSq = TensorPrimitives.SumOfSquares(input);
        float meanSquare = sumSq / n;
        float R = MathF.Sqrt(meanSquare + eps);    // Denominator (root mean square)
        float invR = 1.0f / R;
        float invR3 = 1.0f / (R * R * R);

        // Compute gradGamma = gradOutput * (input / R) elementwise
        // (each gamma_i gets gradient = dL/dy_i * x_i / R)
        TensorPrimitives.Multiply(input, outputGradient, gammaGradient);  // temp: elementwise x_i * gradOut_i
        TensorPrimitives.Multiply(gammaGradient, invR, gammaGradient);  // temp: elementwise x_i * gradOut_i

        float dot = TensorPrimitives.Dot(input, outputGradient);

        // First term: (gamma_i / R) * gradOut_i
        TensorPrimitives.Multiply(outputGradient, gamma, inputGradient);  // gradInput = gradOut * gamma (elementwise)
        TensorPrimitives.Multiply(inputGradient, invR, inputGradient);    // gradInput = gradInput * (1/R)

        // Second term: (gamma_i * x_i / (R^3 * n)) * dot, for each i
        float coeff = invR3 * (dot / n);

        // TensorPrimitives.MultiplyAdd(input, -gamma * coeff, inputGradient, inputGradient);

        for (int i = 0; i < n; ++i)
        {
            inputGradient[i] -= gamma[i] * input[i] * coeff;
        }
    }
}
