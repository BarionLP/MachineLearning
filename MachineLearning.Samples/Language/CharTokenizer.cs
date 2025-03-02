using MachineLearning.Data;
using System.Diagnostics;

namespace MachineLearning.Samples.Language;

public sealed class CharTokenizer(string tokens) : ITokenizer<string>
{
    private readonly string tokens = tokens;
    public int TokenCount => tokens.Length;

    public string GetToken(int index) => tokens[index].ToString();
    public string Decode(IEnumerable<int> tokens) => string.Join("", tokens.Select(index => this.tokens[index]));

    public IEnumerable<int> Tokenize(string data)
    {
        foreach (var c in data)
        {
            var index = tokens.IndexOf(c);
#if DEBUG
            if (index < 0)
            {
                throw new InvalidOperationException($"Unknown token '{c}'");
            }
#endif
            yield return index;
        }
    }

    public int TokenizeSingle(string data)
    {
        Debug.Assert(data.Length == 1);
        var index = tokens.IndexOf(data[0]);
        if (index < 0)
        {
            throw new InvalidOperationException($"Unknown token '{data}'");
        }
        return index;
    }
}
