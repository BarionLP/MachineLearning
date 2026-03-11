// using Ametrin.Serializer;
// using ML.Core.Modules;

// namespace ML.Core.Converters;

// public class ModuleConverter<T> : ISerializationConverter<T> where T : IModule
// {
//     public static T ReadValue(IAmetrinReader reader)
//         => TryReadValue(reader).OrThrow();

//     public static Result<T, DeserializationError> TryReadValue(IAmetrinReader reader)
//     {
//         using var sub = reader.ReadStartObject();
//         var module = AmetrinSerializer.DeserializeDynamic<T>(sub);
//         reader.ReadEndObject();
//         return module;
//     }

//     public static void WriteValue(IAmetrinWriter writer, T value)
//     {
//         writer.WriteStartObject();
        
//         writer.WriteEndObject();
//     }
// }
