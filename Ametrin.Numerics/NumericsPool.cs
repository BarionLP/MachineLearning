using System.Buffers;

namespace Ametrin.Numerics;


public static class NumericsPool
{
    public static Vector RentVector(int size) => Vector.Of(size, ArrayPool<Weight>.Shared.Rent(size));
    public static Matrix RentMatrix(int rows, int columns) => Matrix.Of(rows, columns, ArrayPool<Weight>.Shared.Rent(rows  * columns));

    public static void Return(Vector vector)
    {
        if(vector is VectorSimple simple)
        {
             ArrayPool<Weight>.Shared.Return(simple._storage);
        }
        else
        {
            throw new InvalidOperationException();
        }
    }
    
    public static void Return(Matrix vector)
    {
        if(vector is MatrixFlat simple)
        {
             Return(simple.Storage);
        }
        else
        {
            throw new InvalidOperationException();
        }
    }
}
