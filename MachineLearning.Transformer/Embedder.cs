using System.Diagnostics;
using System.Numerics.Tensors;

namespace MachineLearning.Transformer;

public sealed class Embedder(string Tokens, int EmbeddingDimensions, Weight temperature)
{

    public Matrix EmbeddingMatrix /*W_E*/ { get; } = Matrix.Create(Tokens.Length, EmbeddingDimensions);
    public Matrix UnembeddingMatrix /*W_U*/ { get; } = Matrix.Create(EmbeddingDimensions, Tokens.Length);
    public int EmbeddingDimensions { get; } = EmbeddingDimensions;
    public string Tokens { get; } = Tokens;

    public Matrix Embedd(string input)
    {
        var result = Matrix.Create(input.Length, EmbeddingDimensions);
        for (int i = 0; i < input.Length; i++)
        {
            GetEmbeddingRef(input[i]).CopyTo(result.RowRef(i));
            // TODO: encode position (sinusoidal or learned positional encoding)
        }
        return result;
    }

    private Vector GetEmbeddingRef(char token)
    {
        var tokenIdx = Tokens.IndexOf(token);
        
        if (tokenIdx == -1)
        {
            throw new ArgumentException($"Unkown token: '{token}'!  Valid tokens: [{Tokens}]", nameof(token));
        }
        return EmbeddingMatrix.RowRef(tokenIdx);
    }

    public char Unembed(Vector vector)
    {
        Debug.Assert(vector.Count == EmbeddingDimensions);

        var logits = vector.Multiply(UnembeddingMatrix);

        Debug.Assert(logits.Count == Tokens.Length);

        var max = logits.Max();
        logits.SubtractPointwiseInPlace(max);
        logits.DivideInPlace(temperature);
        var probabilities = logits.SoftMax();
        return Tokens[GetWeightedRandomIndex(probabilities)];

        static int GetWeightedRandomIndex(Vector probabilities)
        {
            var value = Random.Shared.NextDouble();
            for (int i = 0; i < probabilities.Count; i++)
            {
                value -= probabilities[i];
                if (value < 0)
                    return i;
            }
            return probabilities.Count - 1;
        }
    }
}
