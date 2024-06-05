namespace BarionGPT;

public sealed class AttentionHead(ModelInfo info)
{
    public ModelInfo Info { get; } = info;
    public DenseMatrix QueryMatrix = DenseMatrix.CreateRandom(info.QueryDimensions, info.EmbeddingDimensions, info.InitialDistribution);
    public DenseMatrix KeyMatrix = DenseMatrix.CreateRandom(info.QueryDimensions, info.EmbeddingDimensions, info.InitialDistribution);

    public DenseMatrix ValueDownMatrix = DenseMatrix.CreateRandom(info.QueryDimensions, info.EmbeddingDimensions, info.InitialDistribution);
    public DenseMatrix ValueUpMatrix = DenseMatrix.CreateRandom(info.EmbeddingDimensions, info.QueryDimensions, info.InitialDistribution);

    private double QueryDimensionsRoot = Math.Sqrt(info.QueryDimensions);
    public DenseMatrix GetEmbeddingDelta(DenseMatrix input)
    {
        var queryEmbedding = DenseMatrix.Create(Info.QueryDimensions, input.ColumnCount, 0); // query of each token (column)
        var keyEmbedding = DenseMatrix.Create(Info.QueryDimensions, input.ColumnCount, 0); // key of each token (column)
        var valueVectors = DenseMatrix.Create(Info.EmbeddingDimensions, input.ColumnCount, 0); // value of each token (column)
        for(int i = 0; i < input.ColumnCount; i++)
        {
            //multiply each token vector by the QueryMatrix
            var query = QueryMatrix.Multiply(input.Column(i));
            queryEmbedding.SetColumn(i, query);

            //multiply each token vector by the KeyMatrix
            var key = KeyMatrix.Multiply(input.Column(i));
            keyEmbedding.SetColumn(i, key);

            //calculate the change to a token vector v this token vector would produce if this token attends to v
            //to be more efficient the change gets projected into the query space and back to the embedding space
            var valueDown = ValueDownMatrix.Multiply(input.Column(i));
            var valueUp = ValueUpMatrix.Multiply(valueDown);
            valueVectors.SetColumn(i, valueUp);
        }

        // the dot Product of each key with each query defines how much the token of the query should be affected by the token of the key
        // high dot products mean the key token "attends to" the query token
        var attentionPattern = DenseMatrix.Create(input.ColumnCount, input.ColumnCount, 0);
        for(int column = 0; column < input.ColumnCount; column++)
        {
            var vector = DenseVector.Create(input.ColumnCount, 0);
            for(int row = 0; row < input.ColumnCount; row++)
            {
                //force attendance to zero for earlier tokes, allows training on the whole sentence instead just the next word (can be disabled after training)
                vector[row] = row > column ? double.NegativeInfinity : keyEmbedding.Column(row).DotProduct(queryEmbedding.Column(column)) / QueryDimensionsRoot;
            }
            // softmax each column to get a percentage distribution on how much each key token should affect each query token
            attentionPattern.SetColumn(column, vector.Softmax());
        }

        //calculate the delta per token vector by adding up the product of each value vector and its attendance
        var deltaMatrix = DenseMatrix.Create(Info.EmbeddingDimensions, input.ColumnCount, 0);
        for(int column = 0; column < input.ColumnCount; column++)
        {
            Vector<double> vector = DenseVector.Create(Info.EmbeddingDimensions, 0);
            for(int row = 0; row < input.ColumnCount; row++)
            {
                vector += valueVectors.Column(row) * attentionPattern[row, column];
            }
            deltaMatrix.SetColumn(column, vector);
        }

        return deltaMatrix;
    }
}
