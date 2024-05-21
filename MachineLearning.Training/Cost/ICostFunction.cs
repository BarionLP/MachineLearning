namespace MachineLearning.Training.Cost;

public interface ICostFunction
{
    public Weight Cost(Weight output, Weight expected);
    public Weight Derivative(Weight output, Weight expected);
    public void Derivative(Vector output, Vector expected, Vector result) {
        for(int i = 0; i < result.Count; i++) {
            result[i] = Derivative(output[i], expected[i]);
        }
    }
    public Vector Derivative(Vector output, Vector expected) {
        var result = Vector.Create(output.Count);
        Derivative(output, expected, result);
        return result;
    }

    public Weight TotalCost(Vector output, Vector expected)
    {
        //if (output.Count != expected.Count) throw new ArgumentException("Output and Expected length didn't match");
        var totalCost = 0.0;

        foreach (var i in ..output.Count)
        {
            totalCost += Cost(output[i], expected[i]);
        }

        return totalCost;
    }
}
