namespace BarionGPT; 

public sealed class Unembedder(ModelInfo info) {
    public readonly ModelInfo Info = info;
    // one row per known token (it's unembedding vector)
    public readonly DenseMatrix UnembeddingMatrix = DenseMatrix.CreateRandom(info.TokenCount, info.EmbeddingDimensions, info.InitialDistribution);

    public char Unembed(Matrix<double> input) {
        var logits = UnembeddingMatrix.Multiply(input.Column(input.ColumnCount - 1));
        logits = logits.Divide(Info.Temperature).Softmax();
        return Info.ValidTokens[GetWeightedRandomIndex(logits)];

        static int GetWeightedRandomIndex(Vector<double> weights) {
            var value = Random.Shared.NextDouble();
            for(int i = 0; i < weights.Count; i++) {
                value -= weights[i];
                if(value < 0) return i;
            }
            return weights.Count - 1;
        }
    }
}
