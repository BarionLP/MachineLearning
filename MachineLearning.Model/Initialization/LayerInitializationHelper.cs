namespace MachineLearning.Model.Initialization;

public static class InitializationHelper
{
    public static Weight RandomInNormalDistribution(Random random, Weight mean, Weight standardDeviation)
    {
        var x1 = 1 - random.NextSingle();
        var x2 = 1 - random.NextSingle();

        var y1 = MathF.Sqrt(-2.0f * MathF.Log(x1)) * MathF.Cos(2.0f * MathF.PI * x2);
        return y1 * standardDeviation + mean;
    }
    public static Weight RandomInUniformDistribution(Random random, Weight mean, Weight scale)
    {
        return (random.NextSingle() * 2 * scale) - scale + mean;
    }
}