using System.Buffers;

namespace ML.Core.Evaluation.Cost;

/// <summary>
/// Cross-Entropy Cost Function using SoftMax<br/>
/// requires a linear output<br/>
/// requires expected.Sum() == 1<para/>
/// parts of softmax and cross entropy cancel out in the backwards pass reducing operations, also stabilizes gradients because less divisions
/// </summary>
public sealed class CrossEntropyCostFromLogits : ICostFunction<Vector>
{
    public static readonly CrossEntropyCostFromLogits Instance = new();

    public Weight TotalCost(Vector logits, Vector expected)
    {
        NumericsDebug.AssertSameDimensions(logits, expected);

        using var destinationStorage = ArrayPool<Weight>.Shared.RentNumerics(logits.FlatCount);
        var destination = Vector.OfSize(logits, destinationStorage);

        var maxLogit = logits.Max();
        logits.SubtractPointwiseTo(maxLogit, destination);
        destination.PointwiseExpToSelf();
        var expSum = destination.Sum();

        var logSumExp = maxLogit + Weight.Log(expSum);
        var expectedDotLogits = expected.Dot(logits);
        return -expectedDotLogits + logSumExp;
    }

    public void DerivativeTo(Vector logits, Vector expected, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(logits, expected, destination);

        logits.SoftMaxTo(destination);
        destination.SubtractToSelf(expected);
    }
}