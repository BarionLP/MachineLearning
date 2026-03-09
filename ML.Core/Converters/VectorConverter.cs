using System.Runtime.InteropServices;
using Ametrin.Serializer;

namespace ML.Core.Converters;

public sealed class VectorConverter : ISerializationConverter<Vector>
{
    public static Vector ReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
        => TryReadProperty(reader, name).OrThrow();

    public static Result<Vector, DeserializationError> TryReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
    {
        var bytes = reader.ReadBytesProperty(name);
        Debug.Assert(bytes.Length % sizeof(Weight) is 0);
        var floatSpan = MemoryMarshal.Cast<byte, float>(bytes);
        return Vector.Of([.. floatSpan]);
    }

    public static void WriteProperty(IAmetrinWriter writer, ReadOnlySpan<char> name, Vector value)
    {
        writer.WritePropertyName(name);
        writer.WriteArrayValue(value.AsSpan(), static (v, writer) => writer.WriteSingleValue(v));
    }
}

public sealed class MatrixConverter : ISerializationConverter<Matrix>
{
    public static Matrix ReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
        => TryReadProperty(reader, name).OrThrow();

    public static Result<Matrix, DeserializationError> TryReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
    {
        reader.ReadPropertyName(name);
        using var sub = reader.ReadStartObject();
        var rowCount = reader.ReadInt32Property("RowCount");
        var storage = VectorConverter.ReadProperty(reader, "Storage");
        Debug.Assert(storage.Count % rowCount == 0);
        var columnCount = storage.Count / rowCount;
        reader.ReadEndObject();
        return Matrix.Of(rowCount, columnCount, storage);
    }

    public static void WriteProperty(IAmetrinWriter writer, ReadOnlySpan<char> name, Matrix value)
    {
        writer.WritePropertyName(name);
        writer.WriteStartObject();
        writer.WriteInt32Property("RowCount", value.RowCount);
        VectorConverter.WriteProperty(writer, "Storage", value.Storage);
        writer.WriteEndObject();
    }
}