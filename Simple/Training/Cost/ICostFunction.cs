namespace Simple.Training.Cost;

public interface ICostFunction
{
    public Number Cost(Number output, Number expected);
    public Number Derivative(Number output, Number expected);
    public Number TotalCost(Number[] output, Number[] expected)
    {
        if (output.Length != expected.Length) throw new ArgumentException("Output and Expected length didn't match");
        Number totalCost = 0.0;

        foreach (var i in ..output.Length)
        {
            totalCost += Cost(output[i], expected[i]);
        }

        return totalCost;
    }
}
