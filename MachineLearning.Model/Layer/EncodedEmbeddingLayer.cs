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

    public EncodedEmbeddingLayer(int tokenCount, int contextSize) : this(tokenCount, (int)Math.Log2(tokenCount) + 1, contextSize) { }
    public EncodedEmbeddingLayer(int tokenCount, int embeddingSize, int contextSize)
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

    public Vector Forward(int[] input)
    {
        var output = Vector.Create(OutputNodeCount);
        var outSpan = output.AsSpan();

        foreach (var i in ..input.Length)
        {
            EmbeddingMatrix.RowSpan(input[i]).CopyTo(outSpan.Slice(i * EmbeddingSize, EmbeddingSize));
        }

        return output;
    }

    public Vector Forward(int[] input, ILayerSnapshot _) => Forward(input);
}
