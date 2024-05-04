using System.Diagnostics;

namespace MachineLearning.Training;

/// <summary>
/// Converts data into expected OutputWeights
/// </summary>
/// <typeparam name="TOutput">Network Output type</typeparam>
/// <typeparam name="TData">Network Weight type</typeparam>
public interface IOutputResolver<in TOutput, out TData>
{
    public TData Expected(TOutput output);
}

public sealed class MNISTOutputResolver : IOutputResolver<int, Number[]>
{
    public Number[] Expected(int output)
    {
        return output switch
        {
            0 => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            4 => [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            6 => [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            7 => [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            8 => [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            9 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            _ => throw new UnreachableException()
        };
    }
}