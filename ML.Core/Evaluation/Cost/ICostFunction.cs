namespace ML.Core.Evaluation.Cost;

public interface ICostFunction<TArch>
    where TArch : ITensorLike<TArch>
{
    public Weight TotalCost(TArch output, TArch expected);
    public void DerivativeTo(TArch output, TArch expected, TArch destination);

    public TArch Derivative(TArch output, TArch expected)
    {
        var result = TArch.OfSize(output);
        DerivativeTo(output, expected, result);
        NumericsDebug.AssertValidNumbers(result.AsSpan());
        return result;
    }
}

public interface ICostFunction : ICostFunction<Vector>
{
    internal Weight Cost(Weight output, Weight expected);
    Weight ICostFunction<Vector>.TotalCost(Vector output, Vector expected)
    {
        NumericsDebug.AssertSameDimensions(output, expected);
        var totalCost = 0.0f;

        foreach (var i in ..output.Count)
        {
            totalCost += Cost(output[i], expected[i]);
        }

        return totalCost;
    }

    internal Weight Derivative(Weight output, Weight expected);

    void ICostFunction<Vector>.DerivativeTo(Vector output, Vector expected, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(output, expected, destination);
        for (int i = 0; i < destination.Count; i++)
        {
            destination[i] = Derivative(output[i], expected[i]);
        }
    }
}