using Ametrin.Utils.Transformation;

namespace MachineLearning.Data.Noise;

public sealed class ImageInputNoise : IInputDataNoise<float[]>
{
    public required int Size { get; init; }
    public float NoiseStrength { get; init; } = 0;
    public float NoiseProbability { get; init; } = 0;
    public int MaxShift { get; init; } = 0;
    public float MaxAngle { get; init; } = 0;
    public float MinScale { get; init; } = 1;
    public float MaxScale { get; init; } = 1;
    public Random Random { get; init; } = Random.Shared;

    public float[] Apply(float[] data)
    {
        var transform = new Transform2D
        {
            Scale = Random.NextSingle(MinScale, MaxScale),
            Rotation = Angle.FromDegrees(Random.NextDouble(-MaxAngle, MaxAngle)),
            Offset = (Random.Next(-MaxShift, MaxShift), Random.Next(-MaxShift, MaxShift)),
        };
        var transformed = transform.ApplySmooth(data.Select(f => (double)f).ToArray(), Size);

        foreach(var i in ..transformed.Length)
        {
            transformed[i] += (Random.NextSingle() - 0.5f) * 2 * NoiseStrength;
        }
        return data.Select(d => (float) d).ToArray();
    }
}
