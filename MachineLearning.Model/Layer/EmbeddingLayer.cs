using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class EmbeddingLayer(int tokenCount, int embeddingSize, int contextSize) : IEmbeddingLayer<int[]>
{
    // init randomly with [-0.1; 0.1] or [-0.01; 0.01]
    public Matrix EmbeddingMatrix { get; } = Matrix.Create(tokenCount, embeddingSize);
    public int OutputNodeCount { get; } = contextSize * embeddingSize;
    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;
    public int ContextSize { get; } = contextSize;
    public uint ParameterCount => (uint) EmbeddingMatrix.FlatCount;

    public Vector Forward(int[] input)
    {
        var output = Vector.Create(OutputNodeCount);
        var outSpan = output.AsSpan();

        foreach(var i in ..input.Length)
        {
            GetEmbedding(input[i]).CopyTo(outSpan.Slice(i * EmbeddingSize, EmbeddingSize));
        }

        return output;
    }
    public Vector Forward(int[] input, ILayerSnapshot _) => Forward(input);

    private Span<Weight> GetEmbedding(int index)
    {
        if(index < 0 || index > EmbeddingMatrix.RowCount)
        {
            throw new ArgumentException($"Unknown token: {index}");
        }

        return EmbeddingMatrix.RowSpan(index);
    }

    public sealed class Initializer(Random? random = null) : ILayerInitializer<EmbeddingLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(EmbeddingLayer layer)
        {
            layer.EmbeddingMatrix.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.01f));
        }
    }
}
