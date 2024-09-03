using System.Diagnostics;
using MachineLearning.Domain.Activation;
using MachineLearning.Domain.Numerics.Initialization;

namespace MachineLearning.Transformer;

public sealed class AttentionHead(ModelInfo Info)
{
    public ModelInfo Info { get; } = Info;
    public Matrix KeyWeights /*W_K*/ { get; } = Matrix.Create(Info.KeyQueryDimensions, Info.EmbeddingDimensions);
    public Matrix QueryWeights /*W_Q*/ { get; } = Matrix.Create(Info.KeyQueryDimensions, Info.EmbeddingDimensions);
    public Matrix ValueDownWeights /*W_V*/ { get; } = Matrix.Create(Info.KeyQueryDimensions, Info.EmbeddingDimensions);
    public Matrix ValueUpWeights { get; } = Matrix.Create(Info.EmbeddingDimensions, Info.KeyQueryDimensions);

    private readonly Weight QueryDimensionsSqrt = Math.Sqrt(Info.KeyQueryDimensions);

    public Matrix GetEmbeddingDelta(Matrix input)
    {   
        Debug.Assert(input.RowCount == Info.ContextSize);
        Debug.Assert(input.ColumnCount == Info.EmbeddingDimensions);
        
        var keyEmbedding = Matrix.Create(input.RowCount, Info.KeyQueryDimensions); // key of each token (row)
        var queryEmbedding = Matrix.Create(input.RowCount, Info.KeyQueryDimensions); // query of each token (row)
        var valueVectors = Matrix.Create(input.RowCount, Info.EmbeddingDimensions); // value of each token (row)

        // calculate key and query matricies
        input.MultiplyRowwise(KeyWeights, keyEmbedding);
        input.MultiplyRowwise(QueryWeights, queryEmbedding);
        
        var valueDown = Vector.Create(Info.KeyQueryDimensions);
        for(int i = 0; i < input.RowCount; i++)
        {
            //calculate the change to a token vector v this token vector would produce if this token attends to v
            //to be more efficient the change gets projected into the query space and back to the embedding space
            ValueDownWeights.Multiply(input.RowRef(i), valueDown);
            ValueUpWeights.Multiply(valueDown, valueVectors.RowRef(i));
            // the following can be done with the down projection and projected back up to save on computation (inside the attention block).
        }

        // the dot product of each query with each key defines how much the token of the query should be affected by the token of the key
        // high dot products mean the key token "attends to" the query token
        // rows are the queries and columns the keys
        var attentionPattern = Matrix.CreateSquare(input.RowCount);
        for(int queryIndex = 0; queryIndex < input.RowCount; queryIndex++)
        {
            var vector = attentionPattern.RowRef(queryIndex);
            for(int keyIndex = 0; keyIndex < input.RowCount; keyIndex++)
            {
                //force attendance to 0 (-infty before activation) for earlier tokes, allows training on the whole sentence instead just the next word (can be disabled after training)
                vector[keyIndex] = keyIndex > queryIndex ? double.NegativeInfinity : keyEmbedding.RowRef(keyIndex).Dot(queryEmbedding.RowRef(queryIndex)) / QueryDimensionsSqrt; // divide by sqrt(QueryDimensions) for numerical stability
            }
            // softmax each row to get a percentage distribution on how much each key token should affect each query token
            vector.SoftMaxInPlace();
        }

        //Console.WriteLine(attentionPattern);

        var deltaMatrix = Matrix.Create(input.RowCount, Info.EmbeddingDimensions);
        for (int tokenIndex = 0; tokenIndex < input.RowCount; tokenIndex++)
        {
            var deltaVector = deltaMatrix.RowRef(tokenIndex);
            for (int deltaIndex = 0; deltaIndex < input.RowCount; deltaIndex++)
            {
                var valueVector = valueVectors.RowRef(deltaIndex).Multiply(attentionPattern[tokenIndex, deltaIndex]); // cannot operate on reference because it will be used later again (?)
                deltaVector.AddInPlace(valueVector);
            }
        }

        return deltaMatrix;
    }

    public void Initialize(IInitializer initializer)
    {
        initializer.Initialize(KeyWeights);
        initializer.Initialize(QueryWeights);
        initializer.Initialize(ValueDownWeights);
        initializer.Initialize(ValueUpWeights);
    }

    public int GetWeightsCount() {
        return 
            Info.KeyQueryDimensions * Info.EmbeddingDimensions * 2 + // key & query weights
            Info.KeyQueryDimensions * Info.EmbeddingDimensions + //value down
            Info.EmbeddingDimensions * Info.KeyQueryDimensions //value up
        ;
    }
}