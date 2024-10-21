﻿using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Serialization.Activation;
using System.Collections.Immutable;

namespace MachineLearning.Serialization;

public sealed class GenericModelSerializer(FileInfo fileInfo)
{
    public const string FILE_EXTENSION = ".gmw";
    public const uint FORMAT_VERSION = 1;

    public static readonly Dictionary<Type, (string key, uint version, Func<IModel, BinaryWriter, ErrorState> writer)> ModelSerializers = [];
    public static readonly Dictionary<(string key, uint version), Func<BinaryReader, Result<IModel>>> ModelDeserializers = [];

    public static readonly Dictionary<Type, (string key, uint version, Func<ILayer, BinaryWriter, ErrorState> writer)> LayerSerializers = [];
    public static readonly Dictionary<(string key, uint version), Func<BinaryReader, Result<ILayer>>> LayerDeserializers = [];

    static GenericModelSerializer()
    {
        RegisterModel("slm", 1, SaveSLM, ReadSLM);

        RegisterLayer("simple", 1, SaveSimpleLayer, ReadSimpleLayer);
        RegisterLayer("string", 1, SaveStringLayer, ReadStringLayer);
        RegisterLayer("tokenOut", 1, SaveTokenOutLayer, ReadTokenOutLayer);
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

    public static ErrorState SaveSLM(FeedForwardModel<string, char> model, BinaryWriter writer)
    {
        writer.Write(model.LayerCount);
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
            if (flag.IsFail)
            {
                return flag;
            }
        }

        return null;
    }

    public static Result<FeedForwardModel<string, char>> ReadSLM(BinaryReader reader)
    {
        var layerCount = reader.ReadInt32();
        var layers = new ILayer[layerCount];
        foreach (var i in ..layerCount)
        {
            var layerKey = reader.ReadString();
            var layerVersion = reader.ReadUInt32();

            if (!LayerDeserializers.TryGetValue((layerKey, layerVersion), out var deserializer))
            {
                return new NotImplementedException();
            }

            var result = deserializer(reader);
            if (result.Error is Exception error)
            {
                return error;
            }
            layers[i] = result.OrThrow();
        }

        if (layers[0] is not IEmbeddingLayer<string> inputLayer)
        {
            return new InvalidDataException();
        }

        if (layers[^1] is not IUnembeddingLayer<char> outputLayer)
        {
            return new InvalidDataException();
        }

        return new FeedForwardModel<string, char>
        {
            InputLayer = inputLayer,
            HiddenLayers = layers.Skip(1).Take(layerCount - 2).Cast<SimpleLayer>().ToImmutableArray(),
            OutputLayer = outputLayer,
        };
    }

    public static ErrorState SaveSimpleLayer(SimpleLayer layer, BinaryWriter writer)
    {
        writer.Write(layer.InputNodeCount);
        writer.Write(layer.OutputNodeCount);
        ActivationMethodSerializer.WriteV2(writer, layer.ActivationFunction);


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

    public static Result<SimpleLayer> ReadSimpleLayer(BinaryReader reader)
    {
        var inputNodeCount = reader.ReadInt32();
        var outputNodeCount = reader.ReadInt32();
        var activationMethod = ActivationMethodSerializer.ReadV2(reader);
        var layerBuilder = new LayerFactory(inputNodeCount, outputNodeCount).SetActivationFunction(activationMethod);
        var layer = layerBuilder.Create();

        // decode weights & biases
        foreach (var outputIndex in ..outputNodeCount)
        {
            layer.Biases[outputIndex] = reader.ReadDouble();
            foreach (var inputIndex in ..inputNodeCount)
            {
                layer.Weights[outputIndex, inputIndex] = reader.ReadDouble();
            }
        }

        return layer;
    }

    public static ErrorState SaveStringLayer(StringEmbeddingLayer layer, BinaryWriter writer)
    {
        writer.Write(layer.Tokens);
        writer.Write(layer.ContextSize);
        writer.Write(layer.EmbeddingSize);

        // encode weights & biases
        foreach (var tokenIndex in ..layer.Tokens.Length)
        {
            foreach (var embeddingIndex in ..layer.EmbeddingSize)
            {
                writer.Write(layer.EmbeddingMatrix[tokenIndex, embeddingIndex]);
            }
        }

        return null;
    }

    public static Result<StringEmbeddingLayer> ReadStringLayer(BinaryReader reader)
    {
        var tokens = reader.ReadString();
        var contextSize = reader.ReadInt32();
        var embeddingSize = reader.ReadInt32();
        var layer = new StringEmbeddingLayer(tokens, contextSize, embeddingSize);

        // encode weights & biases
        foreach (var tokenIndex in ..tokens.Length)
        {
            foreach (var embeddingIndex in ..embeddingSize)
            {
                layer.EmbeddingMatrix[tokenIndex, embeddingIndex] = reader.ReadDouble();
            }
        }

        return layer;
    }

    public static ErrorState SaveTokenOutLayer(TokenOutputLayer layer, BinaryWriter writer)
    {
        writer.Write(layer.Tokens);
        writer.Write(layer.WeightedRandom);

        return null;
    }

    public static Result<TokenOutputLayer> ReadTokenOutLayer(BinaryReader reader)
    {
        var tokens = reader.ReadString();
        var weightedRandom = reader.ReadBoolean();

        return new TokenOutputLayer(tokens, weightedRandom);
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
            1 => LoadV1(reader),
            _ => new NotImplementedException(),
        };
    }

    private static Result<IModel> LoadV1(BinaryReader reader)
    {
        var modelKey = reader.ReadString();
        var modelVersion = reader.ReadUInt32();
        if (!ModelDeserializers.TryGetValue((modelKey, modelVersion), out var deserializer))
        {
            return new NotImplementedException();
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
