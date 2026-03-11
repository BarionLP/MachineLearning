// using System.Runtime.InteropServices;
// using Ametrin.Serializer;

// namespace ML.Core.Converters;

// public sealed class VectorConverter : ISerializationConverter<Vector>
// {
//     public static Vector ReadValue(IAmetrinReader reader)
//         => TryReadValue(reader).OrThrow();

//     public static Result<Vector, DeserializationError> TryReadValue(IAmetrinReader reader)
//     {
//         reader.readArray
//         var bytes = reader.TryReadBytesValue().OrThrow();
//         Debug.Assert(bytes.Length % sizeof(Weight) is 0);
//         var floatSpan = MemoryMarshal.Cast<byte, float>(bytes);
//         return Vector.Of([.. floatSpan]);
//     }

//     public static void WriteValue(IAmetrinWriter writer, Vector value)
//     {
//         writer.WriteArrayValue(value.AsSpan(), static (v, writer) => writer.WriteSingleValue(v));
//     }
// }