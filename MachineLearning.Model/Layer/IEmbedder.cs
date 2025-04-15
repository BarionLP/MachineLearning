namespace MachineLearning.Model.Embedding;

// public interface IEmbedder<TInput, TOutput> : IEmbeddingLayer<TInput>, IUnembeddingLayer<TOutput>
// {
//     public Vector Embed(TInput input);
//     public Vector Embed(TInput input, ILayerSnapshot snapshot) => Embed(input);
//     public (TOutput output, Weight confidence) Unembed(Vector input);
//     public (TOutput output, Weight confidence, Vector weights) Unembed(Vector input, ILayerSnapshot snapshot)
//     {
//         var (output, confidence) = Unembed(input);
//         return (output, confidence, input);
//     }

//     int IEmbeddingLayer<TInput>.OutputNodeCount => 0;
//     int IUnembeddingLayer<TOutput>.InputNodeCount => 0;
//     long ILayer.ParameterCount => 0;

//     Vector IEmbeddingLayer<TInput>.Process(TInput input) => Embed(input);
//     Vector IEmbeddingLayer<TInput>.Process(TInput input, ILayerSnapshot snapshot) => Embed(input, snapshot);

//     (TOutput output, Weight confidence) IUnembeddingLayer<TOutput>.Process(Vector input) => Unembed(input);
//     (TOutput output, Weight confidence, Vector weights) IUnembeddingLayer<TOutput>.Process(Vector input, ILayerSnapshot snapshot) => Unembed(input, snapshot);
//     ILayerSnapshot ILayer.CreateSnapshot() => LayerSnapshots.Empty;
// }