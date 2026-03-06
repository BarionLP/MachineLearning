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
        writer.WriteStartObject();
        // TODO: might need to switch bytes on big endian systems;
        writer.WriteBytesProperty(name, MemoryMarshal.AsBytes(value.AsSpan()));
    }
}

public sealed class MatrixConverter : ISerializationConverter<Matrix>
{
    public static Matrix ReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
        => TryReadProperty(reader, name).OrThrow();

    public static Result<Matrix, DeserializationError> TryReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
    {
        var bytes = reader.ReadBytesProperty(name);
        Debug.Assert(bytes.Length % sizeof(Weight) is 0);
        var floatSpan = MemoryMarshal.Cast<byte, float>(bytes);
        return Matrix.Of([.. floatSpan]);
    }

    public static void WriteProperty(IAmetrinWriter writer, ReadOnlySpan<char> name, Matrix value)
    {
        // TODO: might need to switch bytes on big endian systems;
        writer.WriteBytesProperty(name, MemoryMarshal.AsBytes(value.AsSpan()));
    }
}

[GenerateSerializer(AllProperties: true)]
public partial struct MatrixDTO(Matrix matrix)
{
    public int ColumnCount { get; } = matrix.ColumnCount;
    public int RowCount { get; } = matrix.RowCount;
}
