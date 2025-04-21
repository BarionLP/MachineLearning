using System.Numerics.Tensors;
using MachineLearning.Model.Attributes;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Serialization;
using MachineLearning.Training.Attributes;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;
using MachineLearning.Training.Optimization.Adam;


namespace MachineLearning.Mamba;

[GeneratedLayer, LayerSerializer("emb", 2), GenerateOptimizers]
public sealed partial class EmbeddingLayer : ILayer<int[], Matrix, EmbeddingLayer.Snapshot>
{
    [Parameter] public int ContextSize { get; }
    // init randomly with [-0.1; 0.1] or [-0.01; 0.01]
    [Weights] public Matrix EmbeddingMatrix { get; }

    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;
    public int TokenCount => EmbeddingMatrix.RowCount;

    public EmbeddingLayer(int tokenCount, int contextSize, int embeddingSize)
        : this(contextSize, Matrix.Create(tokenCount, embeddingSize)) { }

    public Matrix Forward(int[] input, Snapshot snapshot)
    {
        Debug.Assert(input.Length <= ContextSize);
        snapshot.Input = input;

        foreach (var i in ..input.Length)
        {
            GetEmbedding(input[i]).CopyTo(snapshot.Output.RowSpan(i));
        }

        return snapshot.Output.Rows(..input.Length);
    }

    private Span<Weight> GetEmbedding(int index)
    {
        if (index < 0 || index >= EmbeddingMatrix.RowCount)
        {
            throw new ArgumentException($"Unknown token: {index}");
        }

        return EmbeddingMatrix.RowSpan(index);
    }

    public void Backward(Matrix outputGradients, Snapshot snapshot, Gradients gradients)
    {
        foreach (var i in ..snapshot.Input.Length)
        {
            var token = snapshot.Input[i];
            var embeddingGradient = gradients.EmbeddingMatrix.RowSpan(token);
            TensorPrimitives.Add(embeddingGradient, outputGradients.RowSpan(i), embeddingGradient);
        }
    }

    partial class Snapshot
    {
        public int[] Input { get; set; } = [];
        public Matrix Output { get; } = Matrix.Create(layer.ContextSize, layer.EmbeddingSize);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<EmbeddingLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(EmbeddingLayer layer)
        {
            var limit = Weight.Sqrt(6 / (Weight)(layer.TokenCount + layer.EmbeddingSize));
            layer.EmbeddingMatrix.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0, limit));
        }
    }
}
