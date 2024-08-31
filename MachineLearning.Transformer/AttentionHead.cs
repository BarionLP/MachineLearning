using System.Diagnostics;
using MachineLearning.Domain.Activation;

namespace MachineLearning.Transformer;

public sealed class AttentionHead(ModelInfo info)
{
    public ModelInfo Info { get; } = info;
    public Matrix QueryWeights { get; } = Matrix.Create(info.QueryDimensions, info.EmbeddingDimensions);
    public Matrix KeyWeights { get; } = Matrix.Create(info.QueryDimensions, info.EmbeddingDimensions);
    public Matrix ValueDownWeights { get; } = Matrix.Create(info.QueryDimensions, info.EmbeddingDimensions);
    public Matrix ValueUpWeights { get; } = Matrix.Create(info.EmbeddingDimensions, info.QueryDimensions);

    private readonly Weight QueryDimensionsRoot = Math.Sqrt(info.QueryDimensions);

    public Matrix GetEmbeddingDelta(Matrix input)
    {   
        Debug.Assert(input.RowCount == Info.ContextSize);
        Debug.Assert(input.ColumnCount == Info.EmbeddingDimensions);
        
        var queryEmbedding = Matrix.Create(input.RowCount, Info.QueryDimensions); // query of each token (row)
        var keyEmbedding = Matrix.Create(input.RowCount, Info.QueryDimensions); // key of each token (row)
        var valueVectors = Matrix.Create(input.RowCount, Info.EmbeddingDimensions); // value of each token (row)

        input.MultiplyRowwise(QueryWeights, queryEmbedding);
        input.MultiplyRowwise(KeyWeights, keyEmbedding);
        
        
        var valueDown = Vector.Create(Info.QueryDimensions);
        for(int i = 0; i < input.RowCount; i++)
        {
            //calculate the change to a token vector v this token vector would produce if this token attends to v
            //to be more efficient the change gets projected into the query space and back to the embedding space
            ValueDownWeights.Multiply(input.Row(i), valueDown);
            ValueUpWeights.Multiply(valueDown, valueVectors.Row(i));
        }

        // the dot Product of each key with each query defines how much the token of the query should be affected by the token of the key
        // high dot products mean the key token "attends to" the query token
        var attentionPattern = Matrix.Create(input.RowCount, input.RowCount);
        for(int row_index = 0; row_index < input.RowCount; row_index++)
        {
            var vector = attentionPattern.Row(row_index);
            for(int column_index = 0; column_index < input.RowCount; column_index++)
            {
                //force attendance to 0 (-infty before activation) for earlier tokes, allows training on the whole sentence instead just the next word (can be disabled after training)
                vector[column_index] = column_index < row_index ? double.NegativeInfinity : keyEmbedding.Row(column_index).Dot(queryEmbedding.Row(row_index)) / QueryDimensionsRoot;
            }
            // softmax each row to get a percentage distribution on how much each key token should affect each query token
            SoftmaxActivation.Instance.Activate(vector, vector);
        }

        //Console.WriteLine(attentionPattern);

        var deltaMatrix = Matrix.Create(input.RowCount, Info.EmbeddingDimensions);
        for (int row_index = 0; row_index < input.RowCount; row_index++)
        {
            var deltaVector = deltaMatrix.Row(row_index);
            for (int column_index = 0; column_index < input.RowCount; column_index++)
            {
                var valueVector = valueVectors.Row(column_index).Multiply(attentionPattern[row_index, column_index]); // cannot operate on reference because it will be used later again
                deltaVector.AddInPlace(valueVector);
            }
        }

        return deltaMatrix;
    }
}