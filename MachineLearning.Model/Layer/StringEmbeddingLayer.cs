using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class StringEmbeddingLayer(string tokens, int contextSize, int embeddingSize) : IEmbeddingLayer<string>
{
    // init randomly with [-0.1; 0.1] or [-0.01; 0.01]
    public Matrix EmbeddingMatrix { get; } = Matrix.Create(tokens.Length, embeddingSize);
    public int OutputNodeCount { get; } = contextSize * embeddingSize;
    public string Tokens { get; } = tokens;
    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;
    public int ContextSize { get; } = contextSize;
    public uint ParameterCount => (uint)EmbeddingMatrix.FlatCount;

    public Vector Forward(string input)
    {
        var output = Vector.Create(OutputNodeCount);
        var outSpan = output.AsSpan();

        foreach (var i in ..input.Length)
        {
            GetTokenEmbedding(input[i]).CopyTo(outSpan.Slice(i * EmbeddingSize, EmbeddingSize));
        }

        return output;
    }

    public Span<Weight> GetTokenEmbedding(char token) {
        var tokenIdx = Tokens.IndexOf(token);
        if (tokenIdx < 0)
        {
            throw new ArgumentException($"Unknown token: '{token}'");
        }

        return EmbeddingMatrix.RowSpan(tokenIdx);
    }

    public Vector Forward(string input, ILayerSnapshot rawSnapshot)
    {
        var snapshot = LayerSnapshots.Is<LayerSnapshots.Embedding>(rawSnapshot);
        snapshot.LastInput = input;
        var output = Forward(input);
        //output.CopyTo(snapshot.LastOutput);
        return output;
    }

    public sealed class Initializer(Random? random = null) : ILayerInitializer<StringEmbeddingLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(StringEmbeddingLayer layer)
        {
            layer.EmbeddingMatrix.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0f, 0.01f));
        }
    }
}