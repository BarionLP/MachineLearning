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
        RegisterModel("ffm", 1, SaveFFM, ReadFFM);

        RegisterLayer("simple", 2, SaveSimpleLayer, ReadSimpleLayer);
        //RegisterLayer("string", 1, SaveStringLayer, ReadStringLayer);
        //RegisterLayer("tokenOut", 1, SaveTokenOutLayer, ReadTokenOutLayer);
    }

    public static ErrorState SaveFFM(FeedForwardModel model, BinaryWriter writer)
    {
        writer.Write(model.Layers.Length);
        foreach (var layer in model.Layers)
        {
            if (!LayerSerializers.TryGetValue(layer.GetType(), out var data))
            {
                return new NotImplementedException();
            }

            var (key, subVersion, serializer) = data;

            writer.Write(key);
            writer.Write(subVersion);
            var flag = serializer(layer, writer);
            if (!OptionsMarshall.IsSuccess(flag))
            {
                return flag;
            }
        }

        return default;
    }

    public static Result<FeedForwardModel> ReadFFM(BinaryReader reader)
    {
        var layerCount = reader.ReadInt32();
        var layers = new FeedForwardLayer[layerCount];
        foreach (var i in ..layerCount)
        {
            var layerKey = reader.ReadString();
            var layerVersion = reader.ReadUInt32();

            if (!LayerDeserializers.TryGetValue((layerKey, layerVersion), out var deserializer))
            {
                return new NotImplementedException($"No reader registered for {layerKey} v{layerVersion} layer");
            }

            var result = deserializer(reader);
            if (OptionsMarshall.TryGetError(result, out var error))
            {
                return error;
            }
            layers[i] = result.Where<FeedForwardLayer>().OrThrow();
        }

        return new FeedForwardModel
        {
            Layers = [.. layers],
        };
    }

    public static ErrorState SaveSimpleLayer(FeedForwardLayer layer, BinaryWriter writer)
    {
        writer.Write(layer.InputNodeCount);
        writer.Write(layer.OutputNodeCount);
        ActivationFunctionSerializer.WriteV3(writer, layer.ActivationFunction);


        // encode weights & biases
        foreach (var outputIndex in ..layer.OutputNodeCount)
        {
            writer.Write(layer.Biases[outputIndex]);
            foreach (var inputIndex in ..layer.InputNodeCount)
            {
                writer.Write(layer.Weights[outputIndex, inputIndex]);
            }
        }

        return null;
    }

    public static Result<FeedForwardLayer> ReadSimpleLayer(BinaryReader reader)
    {
        var inputNodeCount = reader.ReadInt32();
        var outputNodeCount = reader.ReadInt32();
        var activationMethod = ActivationFunctionSerializer.ReadV3(reader);
        var layerBuilder = new LayerFactory(inputNodeCount, outputNodeCount).SetActivationFunction(activationMethod);
        var layer = layerBuilder.Create();

        // decode weights & biases
        foreach (var outputIndex in ..outputNodeCount)
        {
            layer.Biases[outputIndex] = reader.ReadSingle();
            foreach (var inputIndex in ..inputNodeCount)
            {
                layer.Weights[outputIndex, inputIndex] = reader.ReadSingle();
            }
        }

        return layer;
    }

    public ErrorState Save(IModel model)
    {
        if (!ModelSerializers.TryGetValue(model.GetType(), out var data))
        {
            return new NotImplementedException();
        }

        var (key, modelVersion, serializer) = data;

        using var stream = fileInfo.Create();
        using var writer = new BinaryWriter(stream);
        writer.Write(FILE_EXTENSION);
        writer.Write(FORMAT_VERSION);
        writer.Write(key);
        writer.Write(modelVersion);

        return serializer(model, writer);
    }

    public Result<IModel> Load()
    {
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
            2 => LoadV2(reader),
            _ => new NotImplementedException($".gmw version {formatVersion} is unsupported"),
        };
    }

    private static Result<IModel> LoadV2(BinaryReader reader)
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
        => ModelDeserializers.Add((key, version), (br) => reader(br).Where<IModel>());
    public static void RegisterModel<TModel>(string key, uint version, Func<TModel, BinaryWriter, ErrorState> writer, Func<BinaryReader, Result<TModel>> reader) where TModel : IModel
    {
        RegisterModelReader(key, version, reader);
        ModelSerializers.Add(typeof(TModel), (key, version, (layer, bw) => writer((TModel)layer, bw)));
    }

    public static void RegisterLayerReader<TLayer>(string key, uint version, Func<BinaryReader, Result<TLayer>> reader) where TLayer : ILayer
        => LayerDeserializers.Add((key, version), (br) => reader(br).Where<ILayer>());
    public static void RegisterLayer<TLayer>(string key, uint version, Func<TLayer, BinaryWriter, ErrorState> writer, Func<BinaryReader, Result<TLayer>> reader) where TLayer : ILayer
    {
        RegisterLayerReader(key, version, reader);
        LayerSerializers.Add(typeof(TLayer), (key, version, (layer, bw) => writer((TLayer)layer, bw)));
    }
}
