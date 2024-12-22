namespace MachineLearning.Data.Noise;

public interface IInputDataNoise<TData>
{
    public TData Apply(TData data);
}

public sealed class NoInputNoise<TData> : IInputDataNoise<TData>
{
    public static readonly NoInputNoise<TData> Instance = new();
    public TData Apply(TData data) => data;
}
