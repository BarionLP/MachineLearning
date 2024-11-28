using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class EncodedEmbeddingLayer : IEmbeddingLayer<ReadOnlySpan<int>>
{
    public int OutputNodeCount => ContextSize * EmbeddingSize;
    public uint ParameterCount => 0;

    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;
    public int ContextSize { get; }

    public Matrix EmbeddingMatrix;

    public EncodedEmbeddingLayer(int highestTokenIndex, int contextSize)
    {
        ContextSize = contextSize;
        int embeddingSize = (int) Math.Log2(highestTokenIndex) + 1;
        EmbeddingMatrix = Matrix.Create(highestTokenIndex, embeddingSize);

        var pattern = 1;
        foreach(var row in ..EmbeddingMatrix.RowCount)
        {
            var embedding = EmbeddingMatrix.RowSpan(row);
            for (var i = 0; i < embedding.Length; i++)
            { 
                embedding[i] = pattern & (1 << i);
            }

            pattern++;
        }
    }

    public Vector Forward(ReadOnlySpan<int> input)
    {
        throw new NotImplementedException();
    }

    public Vector Forward(ReadOnlySpan<int> input, ILayerSnapshot _)
    {
        var output = Vector.Create(OutputNodeCount);
        var outSpan = output.AsSpan();

        foreach (var i in ..input.Length)
        {
            EmbeddingMatrix.RowSpan(input[i]).CopyTo(outSpan.Slice(i * EmbeddingSize, EmbeddingSize));
        }

        return output;
    }
}
