using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Serialization.Activation;

namespace MachineLearning.Serialization;

/// <summary>
/// Binary Serializer for <see cref="SimpleModel"/> and <see cref="EmbeddedModel{TInput, TOutput}"/>
/// </summary>
public sealed class ModelSerializer(FileInfo fileInfo)
{
    public const uint VERSION = 3;

    public ErrorState<Exception> Save<TInput, TOutput>(EmbeddedModel<TInput, TOutput> model) => Save(model.InnerModel);
    public ErrorState<Exception> Save(SimpleModel model)
    {
        using var stream = fileInfo.Create();
        var writer = new BinaryWriter(stream);
        writer.Write(VERSION);
        writer.Write(model.Layers.Length);
        foreach (var layer in model.Layers)
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
        }

        return null;
    }

    public Result<EmbeddedModel<TInput, TOutput>> Load<TInput, TOutput>(IEmbedder<TInput, TOutput> embedder) => Load().Select(model => new EmbeddedModel<TInput, TOutput>(model, embedder));
    public Result<SimpleModel> Load()
    {
        using var stream = fileInfo.OpenRead();
        using var reader = new BinaryReader(stream);
        var version = reader.ReadUInt32();

        return version switch
        {
            3 => LoadV3(reader),
            2 => LoadV2(reader),
            _ => throw new InvalidDataException(),
        };
    }

    private static SimpleModel LoadV3(BinaryReader reader)
    {
        var layerCount = reader.ReadInt32();
        var layers = new SimpleLayer[layerCount];

        foreach (var layerIndex in ..layerCount)
        {
            var inputNodeCount = reader.ReadInt32();
            var outputNodeCount = reader.ReadInt32();
            var activationMethod = ActivationMethodSerializer.ReadV2(reader);
            var layerBuilder = new LayerFactory(inputNodeCount, outputNodeCount).SetActivationFunction(activationMethod);
            layers[layerIndex] = layerBuilder.Create();

            // decode weights & biases
            foreach (var outputIndex in ..outputNodeCount)
            {
                layers[layerIndex].Biases[outputIndex] = reader.ReadDouble();
                foreach (var inputIndex in ..inputNodeCount)
                {
                    layers[layerIndex].Weights[outputIndex, inputIndex] = reader.ReadDouble();
                }
            }
        }

        return new SimpleModel([.. layers]);
    }

    private static SimpleModel LoadV2(BinaryReader reader)
    {
        var layerCount = reader.ReadInt32();
        var layers = new SimpleLayer[layerCount];

        foreach (var layerIndex in ..layerCount)
        {
            var inputNodeCount = reader.ReadInt32();
            var outputNodeCount = reader.ReadInt32();
            var activationMethod = ActivationMethodSerializer.ReadV1(reader);
            var layerBuilder = new LayerFactory(inputNodeCount, outputNodeCount).SetActivationFunction(activationMethod);
            layers[layerIndex] = layerBuilder.Create();

            // decode weights & biases
            foreach (var outputIndex in ..outputNodeCount)
            {
                layers[layerIndex].Biases[outputIndex] = reader.ReadDouble();
                foreach (var inputIndex in ..inputNodeCount)
                {
                    layers[layerIndex].Weights[outputIndex, inputIndex] = reader.ReadDouble();
                }
            }
        }

        return new SimpleModel([.. layers]);
    }
}
