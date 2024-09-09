namespace MachineLearning.Model.Embedding;

public interface IEmbedder<in TInput, TOutput>
{
    public Vector Embed(TInput input);
    public (TOutput output, Weight confidence) Unembed(Vector input);
}
