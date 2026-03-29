using Ametrin.Serializer;

namespace ML.Core.Converters;

public sealed class MatrixConverter : ISerializationConverter<Matrix>
{
    static MatrixConverter()
    {
        AmetrinSerializer.RegisterSerializer<MatrixConverter, Matrix>();
    }

    public static Result<Matrix, DeserializationError> TryReadValue(IAmetrinReader reader)
    {
        using var objectReader = reader.ReadStartObject();
        var rowCount = objectReader.ReadInt32Property("RowCount");
        objectReader.ReadPropertyName("Storage");
        var storage = VectorConverter.ReadValue<VectorConverter, Vector>(objectReader);
        reader.ReadEndObject();
        Debug.Assert(storage.Count % rowCount == 0);
        var columnCount = storage.Count / rowCount;
        return Matrix.Of(rowCount, columnCount, storage);
    }
    
    public static void WriteValue(IAmetrinWriter writer, Matrix value)
    {
        using var objectWriter = writer.WriteStartObject();
        objectWriter.WriteInt32Property("RowCount", value.RowCount);
        objectWriter.WritePropertyName("Storage");
        VectorConverter.WriteValue(objectWriter, value.Storage);
        writer.WriteEndObject();
    }
}