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
        RegisterModel("etm", 1, SaveETM, ReadETM);

        RegisterLayerReader("simple", 2, ReadFeedForwardLayer);
        RegisterLayer("ffl", 1, SaveFeedForwardLayer, ReadFeedForwardLayer);
        RegisterLayer("eel", 1, SaveEncodedEmbeddingLayer, ReadEncodedEmbeddingLayer);
        RegisterLayer("tol", 1, SaveTokenOutputLayer, ReadTokenOutputLayer);
    }

    public static ErrorState SaveFFM(FeedForwardModel model, BinaryWriter writer)
    {
        writer.Write(model.Layers.Length);
        foreach (var layer in model.Layers)
        {
            var flag = SaveLayer(layer, writer);
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
            var result = ReadLayer(reader);
            if (OptionsMarshall.TryGetError(result, out var error))
            {
                return error;
            }
            layers[i] = result.Require<FeedForwardLayer>().OrThrow();
        }

        return new FeedForwardModel
        {
            Layers = [.. layers],
        };
    }

    public static ErrorState SaveETM(EmbeddedModel<int[], int> model, BinaryWriter writer)
    {
        if (model.InputLayer is not EncodedEmbeddingLayer eel)
        {
            return new NotImplementedException("EmbeddedModel<int[], int> only supports EncodedEmbeddingLayer rn");
        }

        var result = SaveEncodedEmbeddingLayer(eel, writer);
        if (!OptionsMarshall.IsSuccess(result))
        {
            return result;
        }

        result = SaveFFM(model.InnerModel, writer);
        if (!OptionsMarshall.IsSuccess(result))
        {
            return result;
        }

        if (model.OutputLayer is not TokenOutputLayer tol)
        {
            return new NotImplementedException("EmbeddedModel<int[], int> only supports TokenOutputLayer rn");
        }

        result = SaveTokenOutputLayer(tol, writer);
        if (!OptionsMarshall.IsSuccess(result))
        {
            return result;
        }

        return default;
    }

    public static Result<EmbeddedModel<int[], int>> ReadETM(BinaryReader reader)
    {
        var input = ReadEncodedEmbeddingLayer(reader);
        if (OptionsMarshall.TryGetError(input, out var error))
        {
            return error;
        }

        var inner = ReadFFM(reader);
        if (OptionsMarshall.TryGetError(inner, out error))
        {
            return error;
        }

        var output = ReadTokenOutputLayer(reader);
        if (OptionsMarshall.TryGetError(output, out error))
        {
            return error;
        }

        return new EmbeddedModel<int[], int>
        {
            InputLayer = input.OrThrow(),
            InnerModel = inner.OrThrow(),
            OutputLayer = output.OrThrow(),
        };
    }

    public static ErrorState SaveFeedForwardLayer(FeedForwardLayer layer, BinaryWriter writer)
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

    public static Result<FeedForwardLayer> ReadFeedForwardLayer(BinaryReader reader)
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
        if (!ModelSerializers.TryGetValue(model.GetType(), out var data))
        {
            return new NotImplementedException($"Saving {model.GetType()} is not implemented");
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

public static class ModelSerializationHelper
{
    public static void WriteMatrixRaw(Matrix matrix, BinaryWriter writer)
    {
        WriteVectorRaw(matrix.Storage, writer);
    }
    public static void WriteMatrix(Matrix matrix, BinaryWriter writer)
    {
        writer.Write(matrix.RowCount);
        writer.Write(matrix.ColumnCount);
        WriteVectorRaw(matrix.Storage, writer);
    }

    public static void WriteVector(Vector vector, BinaryWriter writer)
    {
        writer.Write(vector.Count);
        WriteVectorRaw(vector, writer);
    }
    public static void WriteVectorRaw(Vector vector, BinaryWriter writer)
    {
        foreach (var i in ..vector.Count)
        {
            writer.Write(vector[i]);
        }
    }

    public static Matrix ReadMatrix(BinaryReader reader)
    {
        int rowCount = reader.ReadInt32();
        int columnCount = reader.ReadInt32();
        return ReadMatrixRaw(rowCount, columnCount, reader);
    }
    public static Matrix ReadMatrixRaw(int rowCount, int columnCount, BinaryReader reader)
    {
        return Matrix.Of(rowCount, columnCount, ReadVectorRaw(rowCount * columnCount, reader));
    }

    public static Vector ReadVector(BinaryReader reader)
    {
        var count = reader.ReadInt32();
        return ReadVectorRaw(count, reader);   
    }

    public static Vector ReadVectorRaw(int count, BinaryReader reader)
    {
        var result = Vector.Create(count);
        foreach (var i in ..count)
        {
            result[i] = reader.ReadSingle();
        }
        return result;
    }
}