namespace MachineLearning.Transformer;

public sealed class AttentionHead(ModelInfo info)
{
    public ModelInfo Info { get; } = info;
    public Matrix QueryWeights { get; } = Matrix.Create(info.QueryDimensions, info.EmbeddingDimensions);
    public Matrix KeyWeights { get; } = Matrix.Create(info.QueryDimensions, info.EmbeddingDimensions);
    public Matrix ValueDownWeights { get; } = Matrix.Create(info.QueryDimensions, info.EmbeddingDimensions);
    public Matrix ValueUpWeights { get; } = Matrix.Create(info.QueryDimensions, info.EmbeddingDimensions);

    private readonly Weight QueryDimensionsRoot = Math.Sqrt(info.QueryDimensions);

    public Matrix GetEmbeddingDelta(Matrix input)
    {
        var queryEmbedding = Matrix.Create(input.RowCount, Info.QueryDimensions); // query of each token (row)
        var keyEmbedding = Matrix.Create(input.RowCount, Info.QueryDimensions); // key of each token (row)
        var valueVectors = Matrix.Create(input.RowCount, Info.EmbeddingDimensions); // value of each token (row)

        QueryWeights.Multiply(input, queryEmbedding);
        KeyWeights.Multiply(input, keyEmbedding);
        
        for(int i = 0; i < input.RowCount; i++)
        {
            //calculate the change to a token vector v this token vector would produce if this token attends to v
            //to be more efficient the change gets projected into the query space and back to the embedding space
            var valueDown = ValueDownWeights.Multiply(input.Row(i));
            ValueUpWeights.Multiply(valueDown, valueVectors.Row(i));
        }

        // the dot Product of each key with each query defines how much the token of the query should be affected by the token of the key
        // high dot products mean the key token "attends to" the query token
        var attentionPattern = Matrix.Create(input.RowCount, input.RowCount);
        for(int row_index = 0; row_index < input.RowCount; row_index++)
        {
            var vector = attentionPattern.Row(row_index);
            for(int row = 0; row < input.RowCount; row++)
            {
                vector[row] = row > row_index ? double.NegativeInfinity : keyEmbedding.Row(row).Dot(queryEmbedding.Row(row_index)) / QueryDimensionsRoot;
            }
            // softmax each column to get a percentage distribution on how much each key token should affect each query token
        }
    }
}