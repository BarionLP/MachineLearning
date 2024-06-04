namespace BarionGPT;

public sealed class Embedder(ModelInfo info)
{
    public readonly ModelInfo Info = info;
    //one column per known token (it's embedding vector)
    public readonly DenseMatrix EmbeddingMatrix = DenseMatrix.CreateRandom(info.EmbeddingDimensions, info.TokenCount, info.InitialDistribution);

    public DenseMatrix Embed(string input)
    {
        if(input.Length > Info.ContextSize)
            throw new InvalidDataException("Input larger than maximum context size");

        var resultMatrix = DenseMatrix.Create(Info.EmbeddingDimensions, input.Length, 0);

        for(int i = 0; i < input.Length; i++)
        {
            resultMatrix.SetColumn(i, EmbeddingMatrix.Column(Info.ValidTokens.IndexOf(input[i])));
        }

        return resultMatrix;
    }
}