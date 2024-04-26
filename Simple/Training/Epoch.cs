using System.Collections;

namespace Simple.Training;

public record Epoch<TInput, TOutput>(int BatchCount, IEnumerable<Batch<TInput, TOutput>> Batches) : IEnumerable<Batch<TInput, TOutput>> {
    public Epoch<TInput, TOutput> ApplyNoise(IInputDataNoise<TInput> inputNoise){
        foreach(var batch in Batches){
            batch.ApplyNoise(inputNoise);
        }
        return this;
    }

    public IEnumerator<Batch<TInput, TOutput>> GetEnumerator() => Batches.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
