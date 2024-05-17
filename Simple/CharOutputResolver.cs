using MachineLearning.Training;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

class CharOutputResolver : IOutputResolver<char, Vector<double>> {
    public Vector<double> Expected(char b)
        => Vector.Build.Dense(8, i => ((b & (1 << i)) != 0) ? 1.0 : 0.0);
}