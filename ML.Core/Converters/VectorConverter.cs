using Ametrin.Serializer;

namespace ML.Core.Converters;

public sealed class VectorConverter : ISerializationConverter<Vector>
{
    static VectorConverter()
    {
        AmetrinSerializer.RegisterSerializer<VectorConverter, Vector>();
    }

    public static Result<Vector, DeserializationError> TryReadValue(IAmetrinReader reader)
    {
        return reader.TryReadArrayValue(static reader => reader.TryReadSingleValue()).Map(Vector.Of);
    }

    public static void WriteValue(IAmetrinWriter writer, Vector value)
    {
        writer.WriteArrayValue(value.AsSpan(), static (writer, v) => writer.WriteSingleValue(v));
    }
}