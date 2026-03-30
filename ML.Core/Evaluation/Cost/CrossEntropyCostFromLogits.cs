using System.Buffers;

namespace ML.Core.Evaluation.Cost;

/// <summary>
/// Cross-Entropy Cost Function using SoftMax<br/>
/// requires a linear output<br/>
/// requires expected.Sum() == 1<para/>
/// parts of softmax and cross entropy cancel out in the backwards pass reducing operations, also stabilizes gradients because less divisions
/// </summary>
public sealed class CrossEntropyCostFromLogits : ICostFunction<Vector>, ICostFunction<Matrix>
{
    public static readonly CrossEntropyCostFromLogits Instance = new();

    public Weight TotalCost(Vector logits, Vector expected)
    {
        NumericsDebug.AssertSameDimensions(logits, expected);

        using var destinationStorage = ArrayPool<Weight>.Shared.RentNumerics(logits.FlatCount);
        var probabilities = Vector.OfSize(logits, destinationStorage);

        logits.SoftMaxTo(probabilities);

        return ((ICostFunction<Vector>)CrossEntropyCostFromProbabilities.Instance).TotalCost(probabilities, expected);
    }

    public void DerivativeTo(Vector logits, Vector expected, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(logits, expected, destination);

        logits.SoftMaxTo(destination);
        destination.SubtractToSelf(expected);
    }

    // TODO: probably needs to be done row-wise
    public Weight TotalCost(Matrix output, Matrix expected) => TotalCost(output.Storage, expected.Storage);
    public void DerivativeTo(Matrix output, Matrix expected, Matrix destination) => DerivativeTo(output.Storage, expected.Storage, destination.Storage);
}