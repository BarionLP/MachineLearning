using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class EncodedEmbeddingLayer : IEmbeddingLayer<int[]>
{
    public int OutputNodeCount => ContextSize * EmbeddingSize;
    public long ParameterCount => 0;

    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;
    public int TokenCount => EmbeddingMatrix.RowCount;
    public int ContextSize { get; }

    public Matrix EmbeddingMatrix;

    public EncodedEmbeddingLayer(int tokenCount, int contextSize) : this(tokenCount, contextSize, (int) Math.Log2(tokenCount) + 1) { }
    public EncodedEmbeddingLayer(int tokenCount, int contextSize, int embeddingSize)
    {
        ContextSize = contextSize;
        EmbeddingMatrix = Matrix.Create(tokenCount, embeddingSize);

        var pattern = 0;
        foreach (var row in ..EmbeddingMatrix.RowCount)
        {
            pattern++;
            var embedding = EmbeddingMatrix.RowSpan(row);
            for (var i = 0; i < embedding.Length; i++)
            {
                embedding[i] = (pattern & (1 << i)) >> i;
            }
        }
    }

    public Vector Process(int[] input)
    {
        var output = Vector.Create(OutputNodeCount);
        var outSpan = output.AsSpan();

        foreach (var i in ..input.Length)
        {
            EmbeddingMatrix.RowSpan(input[i]).CopyTo(outSpan.Slice((i + (ContextSize - input.Length)) * EmbeddingSize, EmbeddingSize));
        }

        return output;
    }

    public Vector Process(int[] input, ILayerSnapshot _) => Process(input);

    public ILayerSnapshot CreateSnapshot() => LayerSnapshots.Empty;
}
