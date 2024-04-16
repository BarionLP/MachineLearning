namespace Simple.Network.Embedding;

public interface IEmbedder<in TInput, TData, out TOutput> {
    public TData Embed(TInput input); 
    public TOutput UnEmbed(TData input); 
}
