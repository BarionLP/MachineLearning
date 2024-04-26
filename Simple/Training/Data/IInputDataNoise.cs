using Simple.Training;
using Simple.Training.Data;

namespace Simple;

public interface IInputDataNoise<TData>{
    public TData Apply(TData data);
}

public sealed class NoInputNoise<TData> : IInputDataNoise<TData>{
    public static readonly NoInputNoise<TData> Instance = new();
    public TData Apply(TData data) => data;
}

public sealed class RandomInputNoise(float Strength, Random? Random = null) : IInputDataNoise<double[]>{
    public float Strength { get; } = Strength;
    public Random Random { get; } = Random ?? Random.Shared;

    public double[] Apply(double[] data){
        var result = new double[data.Length];
        foreach(var i in ..data.Length){
            //clamp?
            result[i] = Math.Clamp(data[i] + ((Random.NextDouble()-0.5) * 2 * Strength), 0, 1);
        }

        return result;
    }
}
