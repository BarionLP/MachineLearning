namespace MachineLearning.Serialization;

public static class ModelSerializationHelper
{
    #region Write
    public static void WriteTensor(Tensor tensor, BinaryWriter writer)
    {
        writer.Write(tensor.RowCount);
        writer.Write(tensor.ColumnCount);
        writer.Write(tensor.LayerCount);
        WriteVectorRaw(tensor.Storage, writer);
    }

    public static void WriteMatrix(Matrix matrix, BinaryWriter writer)
    {
        writer.Write(matrix.RowCount);
        writer.Write(matrix.ColumnCount);
        WriteVectorRaw(matrix.Storage, writer);
    }

    public static void WriteVector(Vector vector, BinaryWriter writer)
    {
        writer.Write(vector.Count);
        WriteVectorRaw(vector, writer);
    }

    public static void WriteInt32(int scalar, BinaryWriter writer)
    {
        writer.Write(scalar);
    }
    #endregion

    #region Read
    public static Tensor ReadTensor(BinaryReader reader)
    {
        int rowCount = reader.ReadInt32();
        int columnCount = reader.ReadInt32();
        int layerCount = reader.ReadInt32();
        return ReadTensorRaw(rowCount, columnCount, layerCount, reader);
    }

    public static Matrix ReadMatrix(BinaryReader reader)
    {
        int rowCount = reader.ReadInt32();
        int columnCount = reader.ReadInt32();
        return ReadMatrixRaw(rowCount, columnCount, reader);
    }

    public static Vector ReadVector(BinaryReader reader)
    {
        var count = reader.ReadInt32();
        return ReadVectorRaw(count, reader);
    }

    public static int ReadInt32(BinaryReader reader)
    {
        return reader.ReadInt32();
    }
    #endregion

    #region Raw
    public static Tensor ReadTensorRaw(int rowCount, int columnCount, int layerCount, BinaryReader reader)
    {
        return Tensor.Of(rowCount, columnCount, layerCount, ReadVectorRaw(rowCount * columnCount * layerCount, reader));
    }
    public static Matrix ReadMatrixRaw(int rowCount, int columnCount, BinaryReader reader)
    {
        return Matrix.Of(rowCount, columnCount, ReadVectorRaw(rowCount * columnCount, reader));
    }

    public static Vector ReadVectorRaw(int count, BinaryReader reader)
    {
        var result = Vector.Create(count);
        foreach (var i in ..count)
        {
            result[i] = reader.ReadSingle();
        }
        return result;
    }

    public static void WriteMatrixRaw(Matrix matrix, BinaryWriter writer)
    {
        WriteVectorRaw(matrix.Storage, writer);
    }

    public static void WriteVectorRaw(Vector vector, BinaryWriter writer)
    {
        foreach (var i in ..vector.Count)
        {
            writer.Write(vector[i]);
        }
    }
    #endregion
}