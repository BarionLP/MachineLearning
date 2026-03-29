namespace ML.Core.Data.Noise;

public interface IDataNoise<TData>
{
    public TData Apply(TData data);
}


public sealed class NoDataNoise<TData> : IDataNoise<TData>
{
    public static NoDataNoise<TData> Instance => field ??= new();
    public TData Apply(TData data) => data;
}
