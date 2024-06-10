using System.Collections.Frozen;

namespace MachineLearning.Samples.MNIST;

public sealed class MNISTOutputResolver : IOutputResolver<int>
{
    private readonly FrozenDictionary<int, Vector> _map = new Dictionary<int, Vector>(){
        { 0, Vector.Of([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
        { 1, Vector.Of([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])},
        { 2, Vector.Of([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])},
        { 3, Vector.Of([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])},
        { 4, Vector.Of([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])},
        { 5, Vector.Of([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])},
        { 6, Vector.Of([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])},
        { 7, Vector.Of([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])},
        { 8, Vector.Of([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])},
        { 9, Vector.Of([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])},
    }.ToFrozenDictionary();

    public Vector Expected(int output)
    {
        return _map[output];
    }
}