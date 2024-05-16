using System.Collections.Frozen;
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

public sealed class MNISTOutputResolver : IOutputResolver<int, Vector<double>>
{
    private	readonly FrozenDictionary<int, Vector<double>> _map = new Dictionary<int, Vector<double>>(){
        { 0, Vector.Build.DenseOfEnumerable([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
        { 1, Vector.Build.DenseOfEnumerable([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])},
        { 2, Vector.Build.DenseOfEnumerable([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])},
        { 3, Vector.Build.DenseOfEnumerable([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])},
        { 4, Vector.Build.DenseOfEnumerable([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])},
        { 5, Vector.Build.DenseOfEnumerable([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])},
        { 6, Vector.Build.DenseOfEnumerable([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])},
        { 7, Vector.Build.DenseOfEnumerable([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])},
        { 8, Vector.Build.DenseOfEnumerable([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])},
        { 9, Vector.Build.DenseOfEnumerable([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])},
    }.ToFrozenDictionary();
    
    public Vector<double> Expected(int output)
    {
        return _map[output];
    }
}