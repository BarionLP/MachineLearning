using Ametrin.Utils.Transformation;

namespace MachineLearning.Data.Noise;

public sealed class ImageInputNoise : IInputDataNoise<double[]>
{
    public required int Size { get; init; }
    public double NoiseStrength { get; init; } = 0;
    public double NoiseProbability { get; init; } = 0;
    public int MaxShift { get; init; } = 0;
    public double MaxAngle { get; init; } = 0;
    public double MinScale { get; init; } = 1;
    public double MaxScale { get; init; } = 1;
    public Random Random { get; init; } = Random.Shared;

    public double[] Apply(double[] data)
    {
        var transform = new Transform2D
        {
            Scale = Random.NextDouble(MinScale, MaxScale),
            Rotation = Angle.FromDegrees(Random.NextDouble(-MaxAngle, MaxAngle)),
            Offset = (Random.Next(-MaxShift, MaxShift), Random.Next(-MaxShift, MaxShift)),
        };
        var transformed = transform.ApplySmooth(data, Size);

        foreach(var i in ..transformed.Length)
        {
            transformed[i] += (Random.NextDouble() - 0.5) * 2 * NoiseStrength;
        }
        return transformed;
    }
}
