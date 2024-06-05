using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Serialization.Activation;

namespace MachineLearning.Serialization;

/// <summary>
/// Serializer for <see cref="INetwork{TInput, TData, TOutput, TLayer}"/> using <see cref="double"/> as weight type
/// </summary>
/// <typeparam name="TInput">network input type</typeparam>
/// <typeparam name="TOutput">network output type</typeparam>
/// <typeparam name="TLayer">layer type</typeparam>
public sealed class NetworkSerializer<TInput, TOutput, TLayer>(FileInfo fileInfo) where TLayer : ILayer
{
    public const uint VERSION = 2;

    public ResultFlag Save(INetwork<TInput, TOutput, TLayer> network)
    {
        using var stream = fileInfo.Create();
        var writer = new BinaryWriter(stream);
        writer.Write(VERSION); // version
        writer.Write(network.Layers.Length);
        foreach(var layer in network.Layers)
        {
            writer.Write(layer.InputNodeCount);
            writer.Write(layer.OutputNodeCount);
            ActivationMethodSerializer.Write(writer, layer.ActivationFunction);


            // encode weights & biases
            foreach(var outputIndex in ..layer.OutputNodeCount)
            {
                writer.Write(layer.Biases[outputIndex]);
                foreach(var inputIndex in ..layer.InputNodeCount)
                {
                    writer.Write(layer.Weights[outputIndex, inputIndex]);
                }
            }
        }

        return ResultFlag.Succeeded;
    }

    //TODO: Serialize embedder (is it even possible?!)
    public Result<TNetwork> Load<TNetwork>(IEmbedder<TInput, TOutput> embedder) where TNetwork : INetwork<TInput, TOutput, TLayer>
    {
        using var stream = fileInfo.OpenRead();
        var reader = new BinaryReader(stream);
        var version = reader.ReadUInt32();
        if(version != VERSION)
            throw new InvalidDataException();
        var layerCount = reader.ReadInt32();
        var layers = new TLayer[layerCount];

        foreach(var layerIndex in ..layerCount)
        {
            var inputNodeCount = reader.ReadInt32();
            var outputNodeCount = reader.ReadInt32();
            var activationMethod = ActivationMethodSerializer.Read(reader);
            var layerBuilder = new LayerBuilder<TLayer>(inputNodeCount, outputNodeCount).SetActivationMethod(activationMethod);

            // decode weights & biases
            foreach(var outputIndex in ..outputNodeCount)
            {
                layerBuilder.Biases[outputIndex] = reader.ReadDouble();
                foreach(var inputIndex in ..inputNodeCount)
                {
                    layerBuilder.Weights[outputIndex, inputIndex] = reader.ReadDouble();
                }
            }
            layers[layerIndex] = layerBuilder.Build();
        }

        return (TNetwork) TNetwork.Create(layers, embedder);
    }
}
