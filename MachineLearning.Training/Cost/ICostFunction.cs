namespace MachineLearning.Training.Cost;

public interface ICostFunction
{
    public double Cost(double output, double expected);
    public double Derivative(double output, double expected);
    public double TotalCost(double[] output, double[] expected)
    {
        if (output.Length != expected.Length) throw new ArgumentException("Output and Expected length didn't match");
        double totalCost = 0.0;

        foreach (var i in ..output.Length)
        {
            totalCost += Cost(output[i], expected[i]);
        }

        return totalCost;
    }
}
