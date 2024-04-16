using Ametrin.Utils.Optional;
using Simple.Network;
using Simple.Network.Layer;

namespace Simple;

/// <summary>
/// Serializer for <see cref="INetwork{TInput, TData, TOutput, TLayer}"/> using <see cref="double"/> as weight type
/// </summary>
/// <typeparam name="TInput">network input type</typeparam>
/// <typeparam name="TOutput">network output type</typeparam>
public sealed class NetworkSerializer<TInput, TOutput, TLayer>(Stream stream) : IDisposable where TLayer : ILayer<double>{
    public Stream Stream { get; } = stream;
    private bool isDisposed;
    private readonly bool isStreamOwned = false;
    public NetworkSerializer(FileInfo fileInfo) 
    : this(fileInfo.Open(FileMode.OpenOrCreate)){
        isStreamOwned = true;
    }

    public ResultFlag Save(INetwork<TInput, double, TOutput, TLayer> network){
        using var writer = new BinaryWriter(Stream);
        writer.WriteBigEndian(1u); // version
        writer.WriteBigEndian(network.Layers.Length);
        foreach(var layer in network.Layers){
            writer.WriteBigEndian(layer.InputNodeCount);
            writer.WriteBigEndian(layer.OutputNodeCount);

            // encode weights & biases
            foreach(var outputIndex in ..layer.OutputNodeCount) {
                writer.WriteBigEndian(layer.Biases[outputIndex]);
                foreach(var inputIndex in ..layer.InputNodeCount) {
                    writer.WriteBigEndian(layer.Weights[inputIndex, outputIndex]);
                }
            }
        }

        return ResultFlag.Succeeded;
    }

    public Result<RecordingNetwork<TInput, TOutput>> Load(){
        using var reader = new BinaryReader(Stream);
        var version = reader.ReadUInt32BigEndian();
        var layerCount = reader.ReadInt32BigEndian();
        var layers = new RecordingLayer[layerCount];
        //TODO: properly allow layer serialization
        foreach(var layerIndex in ..layerCount){
            var inputNodeCount = reader.ReadInt32BigEndian();
            var outputNodeCount = reader.ReadInt32BigEndian();
            layers[layerIndex] = new RecordingLayer(inputNodeCount, outputNodeCount);

            // decode weights & biases
            foreach(var outputIndex in ..outputNodeCount) {
                layers[layerIndex].Biases[outputIndex] = reader.ReadDouble();
                foreach(var inputIndex in ..inputNodeCount) {
                    layers[layerIndex].Weights[inputIndex, outputIndex] = reader.ReadDoubleBigEndian();
                }
            }
        }

        return ResultFlag.Failed;
    }

    private void Dispose(bool disposing){
        if (isDisposed) return;
        
        if (disposing){
            if(isStreamOwned) Stream.Dispose();
        }

        isDisposed = true;
    }

    public void Dispose(){
        // cleanup goes in Dispose(bool)
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
