using MachineLearning.Model;
using MachineLearning.Model.Layer;

namespace MachineLearning.Serialization;

public sealed class ModelSerializer(FileInfo fileInfo)
{
    public const string FILE_EXTENSION = ".gmw";
    public const uint FORMAT_VERSION = 2;

    public static readonly Dictionary<Type, (string key, uint version, Func<IModel, BinaryWriter, ErrorState> writer)> ModelSerializers = [];
    public static readonly Dictionary<(string key, uint version), Func<BinaryReader, Result<IModel>>> ModelDeserializers = [];

    public static readonly Dictionary<Type, (string key, uint version, Func<ILayer, BinaryWriter, ErrorState> writer)> LayerSerializers = [];
    public static readonly Dictionary<(string key, uint version), Func<BinaryReader, Result<ILayer>>> LayerDeserializers = [];

    static ModelSerializer()
    {
        RegisterLayer("eel", 1, SaveEncodedEmbeddingLayer, ReadEncodedEmbeddingLayer);
        RegisterLayer("tol", 1, SaveTokenOutputLayer, ReadTokenOutputLayer);
    }

    public static ErrorState SaveEncodedEmbeddingLayer(EncodedEmbeddingLayer layer, BinaryWriter writer)
    {
        writer.Write(layer.TokenCount);
        writer.Write(layer.ContextSize);
        writer.Write(layer.EmbeddingSize);

        return default;
    }

    public static Result<EncodedEmbeddingLayer> ReadEncodedEmbeddingLayer(BinaryReader reader)
    {
        var tokenCount = reader.ReadInt32();
        var contextSize = reader.ReadInt32();
        var embeddingSize = reader.ReadInt32();
        return new EncodedEmbeddingLayer(tokenCount, contextSize, embeddingSize);
    }

    public static ErrorState SaveTokenOutputLayer(TokenOutputLayer layer, BinaryWriter writer)
    {
        writer.Write(layer.TokenCount);
        writer.Write(layer.WeightedRandom);

        return default;
    }

    public static Result<TokenOutputLayer> ReadTokenOutputLayer(BinaryReader reader)
    {
        var tokenCount = reader.ReadInt32();
        var weightedRandom = reader.ReadBoolean();
        return new TokenOutputLayer(tokenCount, weightedRandom);
    }

    public static ErrorState SaveLayer(ILayer layer, BinaryWriter writer)
    {
        if (!LayerSerializers.TryGetValue(layer.GetType(), out var data))
        {
            return new NotImplementedException($"No writer registered for {layer.GetType().Name}");
        }

        var (key, subVersion, serializer) = data;

        writer.Write(key);
        writer.Write(subVersion);
        return serializer(layer, writer);
    }

    public static Result<ILayer> ReadLayer(BinaryReader reader)
    {
        var layerKey = reader.ReadString();
        var layerVersion = reader.ReadUInt32();

        if (!LayerDeserializers.TryGetValue((layerKey, layerVersion), out var deserializer))
        {
            return new NotImplementedException($"No reader registered for {layerKey} v{layerVersion} layer");
        }

        return deserializer(reader);
    }

    public ErrorState Save(IModel model)
    {
        using var stream = fileInfo.Create();
        using var writer = new BinaryWriter(stream);
        writer.Write(FILE_EXTENSION);
        writer.Write(FORMAT_VERSION);

        return SaveModel(model, writer);
    }

    public static ErrorState SaveModel(IModel model, BinaryWriter writer)
    {
        if (!ModelSerializers.TryGetValue(model.GetType(), out var data))
        {
            return new NotImplementedException($"Saving {model.GetType()} is not implemented");
        }

        var (key, modelVersion, serializer) = data;

        writer.Write(key);
        writer.Write(modelVersion);

        return serializer(model, writer);
    }

    public Result<TModel> Load<TModel>() where TModel : IModel => Load().Require<TModel>();
    public Result<IModel> Load()
    {
        if (!fileInfo.Exists)
        {
            return new FileNotFoundException(null, fileInfo.FullName);
        }

        using var stream = fileInfo.OpenRead();
        using var reader = new BinaryReader(stream);

        var fileType = reader.ReadString();
        if (fileType is not FILE_EXTENSION)
        {
            return new InvalidDataException();
        }

        var formatVersion = reader.ReadUInt32();
        return formatVersion switch
        {
            2 => ReadModel(reader),
            _ => new NotImplementedException($".gmw version {formatVersion} is unsupported"),
        };
    }

    public static Result<IModel> ReadModel(BinaryReader reader)
    {
        var modelKey = reader.ReadString();
        var modelVersion = reader.ReadUInt32();
        if (!ModelDeserializers.TryGetValue((modelKey, modelVersion), out var deserializer))
        {
            return new NotImplementedException($"No reader registered for {modelKey} v{modelVersion} model");
        }

        return deserializer(reader);
    }

    public static void RegisterModelReader<TModel>(string key, uint version, Func<BinaryReader, Result<TModel>> reader) where TModel : IModel
        => ModelDeserializers.Add((key, version), (br) => reader(br).Require<IModel>());
    public static void RegisterModel<TModel>(string key, uint version, Func<TModel, BinaryWriter, ErrorState> writer, Func<BinaryReader, Result<TModel>> reader) where TModel : IModel
    {
        RegisterModelReader(key, version, reader);
        ModelSerializers.Add(typeof(TModel), (key, version, (layer, bw) => writer((TModel)layer, bw)));
    }

    public static void RegisterLayerReader<TLayer>(string key, uint version, Func<BinaryReader, Result<TLayer>> reader) where TLayer : ILayer
        => LayerDeserializers.Add((key, version), (br) => reader(br).Require<ILayer>());
    public static void RegisterLayer<TLayer>(string key, uint version, Func<TLayer, BinaryWriter, ErrorState> writer, Func<BinaryReader, Result<TLayer>> reader) where TLayer : ILayer
    {
        RegisterLayerReader(key, version, reader);
        LayerSerializers.Add(typeof(TLayer), (key, version, (layer, bw) => writer((TLayer)layer, bw)));
    }
}
