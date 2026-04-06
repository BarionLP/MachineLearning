using System.IO;
using System.Runtime.CompilerServices;
using Ametrin.Serializer;
using Ametrin.Serializer.Readers;
using Ametrin.Serializer.Writers;
using ML.Core.Modules;

namespace ML.Core.Converters;

public static class ModuleSerializer
{
    public const string FILE_EXTENSION = ".gmw";
    public const uint FORMAT_VERSION = 3;

#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore
    internal static void Init()
    {
        AmetrinSerializer.RegisterSerializer<SequenceModuleConverter<Vector>, SequenceModule<Vector>>();
        AmetrinSerializer.RegisterSerializer<SequenceModuleConverter<Matrix>, SequenceModule<Matrix>>();
        AmetrinSerializer.RegisterSerializer<SequenceModuleConverter<Tensor>, SequenceModule<Tensor>>();
        AmetrinSerializer.RegisterSerializer<EmbeddedModule<int[], Vector, int>>();
        AmetrinSerializer.RegisterSerializer<EmbeddedModule<int[], Matrix, int>>();
    }

    public static void Write(IModule module, FileInfo file)
    {
        using var stream = file.Create();
        using var writer = new AmetrinBinaryWriter(stream);

        writer.WriteStringProperty("$format", FILE_EXTENSION);
        writer.WriteUInt32Property("$version", FORMAT_VERSION);

        AmetrinSerializer.WriteDynamic(writer, module);
        Console.WriteLine($"Module saved to {file}");
    }

    public static T Read<T>(FileInfo file)
    {
        using var stream = file.OpenRead();
        using var reader = new AmetrinBinaryReader(stream);

        var format = reader.ReadStringProperty("$format");
        if (format is not FILE_EXTENSION) throw new InvalidOperationException();
        var version = reader.ReadUInt32Property("$version");
        if (version is not FORMAT_VERSION) throw new InvalidOperationException();

        Console.WriteLine($"Module loaded from {file}");
        return AmetrinSerializer.TryReadDynamic<T>(reader).Or(e => e.Throw<T>());
    }
}
