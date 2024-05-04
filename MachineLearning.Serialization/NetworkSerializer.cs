using Ametrin.Utils.Optional;
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
public sealed class NetworkSerializer<TInput, TOutput, TLayer>(Stream stream) : IDisposable where TLayer : ILayer<double>
{
    public const uint VERSION = 2;
    public Stream Stream { get; } = stream;
    private bool isDisposed;
    private readonly bool isStreamOwned = false;
    public NetworkSerializer(FileInfo fileInfo)
    : this(fileInfo.Open(FileMode.OpenOrCreate, FileAccess.ReadWrite))
    {
        isStreamOwned = true;
    }

    public ResultFlag Save(INetwork<TInput, double, TOutput, TLayer> network)
    {
        var writer = new BinaryWriter(Stream);
        writer.WriteBigEndian(VERSION); // version
        writer.WriteBigEndian(network.Layers.Length);
        foreach (var layer in network.Layers)
        {
            writer.WriteBigEndian(layer.InputNodeCount);
            writer.WriteBigEndian(layer.OutputNodeCount);
            ActivationMethodSerializer<Number>.Write(writer, layer.ActivationMethod);


            // encode weights & biases
            foreach (var outputIndex in ..layer.OutputNodeCount)
            {
                writer.WriteBigEndian(layer.Biases[outputIndex]);
                foreach (var inputIndex in ..layer.InputNodeCount)
                {
                    writer.WriteBigEndian(layer.Weights[inputIndex, outputIndex]);
                }
            }
        }

        return ResultFlag.Succeeded;
    }

    //TODO: Serialize embedder
    public Result<TNetwork> Load<TNetwork>(IEmbedder<TInput, double[], TOutput> embedder) where TNetwork : INetwork<TInput, double, TOutput, TLayer>
    {
        var reader = new BinaryReader(Stream);
        var version = reader.ReadUInt32BigEndian();
        if (version != VERSION) throw new InvalidDataException();
        var layerCount = reader.ReadInt32BigEndian();
        var layers = new TLayer[layerCount];

        foreach (var layerIndex in ..layerCount)
        {
            var inputNodeCount = reader.ReadInt32BigEndian();
            var outputNodeCount = reader.ReadInt32BigEndian();
            var activationMethod = ActivationMethodSerializer<Number>.Read(reader);
            var layerBuilder = new LayerBuilder<TLayer>(inputNodeCount, outputNodeCount).SetActivationMethod(activationMethod);

            // decode weights & biases
            foreach (var outputIndex in ..outputNodeCount)
            {
                layerBuilder.Biases[outputIndex] = reader.ReadDoubleBigEndian();
                foreach (var inputIndex in ..inputNodeCount)
                {
                    layerBuilder.Weights[inputIndex, outputIndex] = reader.ReadDoubleBigEndian();
                }
            }
            layers[layerIndex] = layerBuilder.Build();
        }

        return (TNetwork)TNetwork.Create(layers, embedder);
    }

    private void Dispose(bool disposing)
    {
        if (isDisposed) return;

        if (disposing)
        {
            // managed
            if (isStreamOwned) Stream.Dispose();
        }

        //unmanaged (implement Finalizer if exists)

        isDisposed = true;
    }

    public void Dispose()
    {
        // cleanup goes in Dispose(bool)
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
