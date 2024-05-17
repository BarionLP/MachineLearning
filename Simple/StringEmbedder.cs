using MachineLearning.Model.Embedding;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public sealed class StringEmbedder(int contextSize) : IEmbedder<string, Vector<double>, char> {
    public Vector<double> Embed(string input) {
        var result = Vector.Build.Dense(8 * input.Length);

        for(var ic = 0; ic < input.Length; ic++) {
            var c = input[ic];
            for(int i = 0; i < 8; i++) {
                result[ic * 8 + i] = ((c & (1 << i)) != 0) ? 1.0 : 0.0;
            }
        }

        return PadLeft(result, contextSize * 8);
    }
    public static Vector<double> PadLeft(Vector<double> vector, int totalWidth, double paddingValue = 0.0) {
        if(vector.Count >= totalWidth)
            return vector;

        var paddedVector = Vector.Build.Dense(totalWidth, paddingValue);
        paddedVector.SetSubVector(totalWidth - vector.Count, vector.Count, vector);
        return paddedVector;
    }

    public char UnEmbed(Vector<double> input) {
        if(input.Count != 8)
            throw new ArgumentException("Input length must be 8.");

        byte result = 0;

        for(int i = 0; i < 8; i++) {
            byte bit = (input[i] >= 0.5) ? (byte) 1 : (byte) 0;
            result |= (byte) (bit << i);
        }

        return (char) result;
    }
}