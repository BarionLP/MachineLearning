// using Ametrin.Serializer;

// namespace ML.Core.Converters;

// public sealed class MatrixConverter : ISerializationConverter<Matrix>
// {
//     public static Matrix ReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
//         => TryReadProperty(reader, name).OrThrow();

//     public static Result<Matrix, DeserializationError> TryReadProperty(IAmetrinReader reader, ReadOnlySpan<char> name)
//     {
//         reader.ReadPropertyName(name);
//         using var sub = reader.ReadStartObject();
//         var rowCount = sub.ReadInt32Property("RowCount");
//         var storage = VectorConverter.ReadProperty(sub, "Storage");
//         Debug.Assert(storage.Count % rowCount == 0);
//         var columnCount = storage.Count / rowCount;
//         reader.ReadEndObject();
//         return Matrix.Of(rowCount, columnCount, storage);
//     }

//     public static void WriteProperty(IAmetrinWriter writer, ReadOnlySpan<char> name, Matrix value)
//     {
//         writer.WritePropertyName(name);
//         writer.WriteStartObject();
//         writer.WriteInt32Property("RowCount", value.RowCount);
//         VectorConverter.WriteProperty(writer, "Storage", value.Storage);
//         writer.WriteEndObject();
//     }
// }