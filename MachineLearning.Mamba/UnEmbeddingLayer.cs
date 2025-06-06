using MachineLearning.Model.Activation;
using MachineLearning.Model.Attributes;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Serialization;
using MachineLearning.Training.Attributes;

namespace MachineLearning.Mamba;

[GeneratedLayer(OutputGradientType = typeof(Matrix)), LayerSerializer("unemb", 2), GenerateOptimizers]
public sealed partial class UnEmbeddingLayer : ILayer<Matrix, (Vector, int), UnEmbeddingLayer.Snapshot>
{
    [Parameter] public int ContextSize { get; }
    [Weights] public Matrix UnEmbeddingMatrix { get; }
    public int TokenCount => UnEmbeddingMatrix.RowCount;

    public int EmbeddingSize => UnEmbeddingMatrix.ColumnCount;

    public UnEmbeddingLayer(int tokenCount, int contextSize, int embeddingSize)
        : this(contextSize, Matrix.Create(tokenCount, embeddingSize)) { }

    public (Matrix, int) Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.RowCount <= ContextSize);
        Debug.Assert(input.ColumnCount == EmbeddingSize);

        snapshot.Input = input;

        foreach (var i in ..snapshot.SequenceLength)
        {
            UnEmbeddingMatrix.MultiplyTo(input.RowRef(i), snapshot.WeightedInput);
            SoftMaxActivation.Instance.ActivateTo(snapshot.WeightedInput, snapshot.Output.RowRef(i));
        }

        return (snapshot.Output.Rows(..snapshot.SequenceLength), snapshot.Output.RowRef(snapshot.SequenceLength - 1).MaximumIndex());
    }

    public void Backward(Matrix outputGradients, Snapshot snapshot, Gradients gradients)
    {
        Debug.Assert(outputGradients.ColumnCount == TokenCount);
        Debug.Assert(outputGradients.RowCount == snapshot.SequenceLength);

        // this would be neccecary without CrossEntropyFromSoftmaxLoss (not sure if it is correct)
        // var tmp = Vector.Create(outputGradient.Count);
        // SoftMaxActivation.Instance.DerivativeTo(snapshot.WeightedInput, tmp);
        // tmp.PointwiseMultiplyToSelf(outputGradient);

        // y = W * v
        // dy = y - expected // because CrossEntropy and Softmax cancel out
        // => dW += v * dy
        // => dv += W^T * dy

        foreach (var i in ..snapshot.SequenceLength)
        {
            VectorHelper.MultiplyToMatrixAddTo(outputGradients.RowRef(i), snapshot.Input.RowRef(i), gradients.UnEmbeddingMatrix);
            UnEmbeddingMatrix.MultiplyTransposedTo(outputGradients.RowRef(i), snapshot.GradientInput.RowRef(i));
        }
    }

    partial class Snapshot
    {
        public int SequenceLength => Input.RowCount;
        public Matrix Input { get; set; } = null!;
        public Vector WeightedInput { get; } = Vector.Create(layer.TokenCount);
        public Matrix Output { get; } = Matrix.Create(layer.ContextSize, layer.TokenCount);

        public Matrix GradientInput { get; } = Matrix.Create(layer.ContextSize, layer.EmbeddingSize);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<UnEmbeddingLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(UnEmbeddingLayer layer)
        {
            var limit = Weight.Sqrt(6 / (Weight)(layer.TokenCount + layer.EmbeddingSize));
            layer.UnEmbeddingMatrix.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0, limit));
        }
    }
}
