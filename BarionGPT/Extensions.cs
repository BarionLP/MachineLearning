namespace BarionGPT;

public static class Extensions {
    public static void InplaceSoftMax(this Vector<double> input, double temperature = 4) {
        //var max = input.Maximum();
        //input.MapInplace(x => MathF.Exp(x - max)); // Shift values for numerical stability
        var sumExp = input.Sum();
        input.MapInplace(f => f / sumExp);
    }
    
    public static Vector<double> Softmax(this Vector<double> vector) {
        var exp = vector.PointwiseExp();
        var sumExp = exp.Sum();
        return exp.Divide(sumExp);
    }

    public static Matrix<double> RowSoftmax(this Matrix<double> matrix) {
        var expMatrix = matrix.PointwiseExp();
        var sumExpMatrix = expMatrix.RowSums();
        for(int i = 0; i < matrix.RowCount; i++) {
            expMatrix.SetRow(i, expMatrix.Row(i).Divide(sumExpMatrix[i]));
        }
        return expMatrix;
    }
}
