namespace MachineLearning.Model.Embedding;

public interface IEmbedder<in TInput, out TOutput>
{
    public Vector Embed(TInput input);
    public TOutput UnEmbed(Vector input);
}
