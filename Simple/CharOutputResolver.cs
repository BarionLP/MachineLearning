using MachineLearning.Training;

class CharOutputResolver : IOutputResolver<char> {
    public Vector Expected(char b)
        => Vector.Of(Enumerable.Range(0, 8).Select(i => ((b & (1 << i)) != 0) ? 1.0 : 0.0).ToArray());
}