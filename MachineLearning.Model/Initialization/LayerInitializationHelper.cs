namespace MachineLearning.Model.Initialization;

public static class InitializationHelper
{
    public static double RandomInNormalDistribution(Random random, double mean, double standardDeviation)
    {
        var x1 = 1 - random.NextDouble();
        var x2 = 1 - random.NextDouble();

        var y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
        return y1 * standardDeviation + mean;
    }
}