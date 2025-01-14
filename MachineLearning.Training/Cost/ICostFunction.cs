namespace MachineLearning.Training.Cost;

public interface ICostFunction
{
    public Weight Cost(Weight output, Weight expected);
    public Weight Derivative(Weight output, Weight expected);
    public void DerivativeTo(Vector output, Vector expected, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(output, expected, destination);
        for(int i = 0; i < destination.Count; i++)
        {
            destination[i] = Derivative(output[i], expected[i]);
        }
    }
    public Vector Derivative(Vector output, Vector expected)
    {
        var result = Vector.Create(output.Count);
        DerivativeTo(output, expected, result);
        return result;
    }

    public Weight TotalCost(Vector output, Vector expected)
    {
        NumericsDebug.AssertSameDimensions(output, expected);
        var totalCost = 0.0f;

        foreach(var i in ..output.Count)
        {
            totalCost += Cost(output[i], expected[i]);
        }

        return totalCost;
    }
}
