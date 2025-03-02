namespace MachineLearning.Data;

public interface ITokenizer<TData>
{
    public int TokenCount { get; }
    public IEnumerable<int> Tokenize(TData data);
    public int TokenizeSingle(TData data);
    public TData GetToken(int data);
    public string Decode(IEnumerable<int> tokens);

}
