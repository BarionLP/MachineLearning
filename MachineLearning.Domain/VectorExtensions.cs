namespace MachineLearning.Domain;

public static class VectorExtensions
{
    public static void SubtractInPlace(this Vector<double> vectorA, Vector<double> vectorB){
        vectorA.Subtract(vectorB, vectorA);
    }
    public static void AddInPlace(this Vector<double> vectorA, Vector<double> vectorB){
        vectorA.Add(vectorB, vectorA);
    }
    public static void PointwiseMultiplyInPlace(this Vector<double> vectorA, Vector<double> vectorB){
        vectorA.PointwiseMultiply(vectorB, vectorA);
    }
}
public static class MatrixExtensions
{
    public static void SubtractInPlace(this Matrix<double> matrixA, Matrix<double> matrixB){
        matrixA.Subtract(matrixB, matrixA);
    }
    public static void AddInPlace(this Matrix<double> matrixA, Matrix<double> matrixB){
        matrixA.Add(matrixB, matrixA);
    }
}
