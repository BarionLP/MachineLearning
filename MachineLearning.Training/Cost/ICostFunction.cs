namespace MachineLearning.Training.Cost;

public interface ICostFunction
{
    public double Cost(double output, double expected);
    public double Derivative(double output, double expected);
    public Vector<double> Derivative(Vector<double> output, Vector<double> expected)
        => Vector<double>.Build.Dense(output.Count, i => Derivative(output[i], expected[i]));
    public double TotalCost(Vector<double> output, Vector<double> expected)
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
