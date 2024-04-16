using Ametrin.Utils.Optional;
using Simple.Network;
using Simple.Network.Layer;

namespace Simple;

/// <summary>
/// Serializer for <see cref="INetwork{TInput, TData, TOutput, TLayer}"/> using <see cref="Number"/> as weight type
/// </summary>
/// <typeparam name="TInput">network input type</typeparam>
/// <typeparam name="TOutput">network output type</typeparam>
public sealed class NetworkSerializer<TInput, TOutput>(Stream stream) : IDisposable{
    public Stream Stream { get; } = stream;
    private bool isDisposed;
    private readonly bool isStreamOwned = false;
    public NetworkSerializer(FileInfo fileInfo) 
    : this(fileInfo.Open(FileMode.OpenOrCreate)){
        isStreamOwned = true;
    }

    public ResultFlag Save(RecordingNetwork<TInput, TOutput> network){
        using var writer = new BinaryWriter(Stream);
        writer.Write(1u); // version
        writer.Write(network.Layers.Length);
        foreach(var item in network.Layers){
            writer.Write(item.InputNodeCount);
            writer.Write(item.OutputNodeCount);
            // encode weigths
            foreach(var inputIndex in ..item.InputNodeCount){
                
            }
        }

        return ResultFlag.Succeeded;
    }

    public Result<RecordingNetwork<TInput, TOutput>> Load(){
        using var reader = new BinaryReader(Stream);
        var version = reader.ReadUInt32();
        var layerCount = reader.ReadInt32();
        var layers = new RecordingLayer[layerCount];
        //TODO: properly allow layer serialization
        foreach(var layerIndex in ..layerCount){
            var inputNodeCount = reader.ReadInt32();
            var outputNodeCount = reader.ReadInt32();
            layers[layerIndex] = new RecordingLayer(inputNodeCount, outputNodeCount);
        }
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
