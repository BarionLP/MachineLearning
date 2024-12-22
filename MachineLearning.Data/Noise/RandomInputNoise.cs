namespace MachineLearning.Data.Noise;

public sealed class RandomInputNoise(float Strength, Random? Random = null) : IInputDataNoise<float[]>
{
    public float Strength { get; } = Strength;
    public Random Random { get; } = Random ?? Random.Shared;

    public float[] Apply(float[] data)
    {
        var result = new float[data.Length];
        foreach(var i in ..data.Length)
        {
            result[i] = float.Clamp(data[i] + (Random.NextSingle() - 0.5f) * 2 * Strength, 0, 1);
        }

        return result;
    }
}
