using MachineLearning.Data;
using System.Collections.Frozen;

namespace MachineLearning.Samples.Language;

public sealed class StringTokenizer(string[] tokens) : ITokenizer<string>
{
    private readonly FrozenDictionary<string, int> tokens = tokens.Select((token, i) => new KeyValuePair<string, int>(token, i)).ToFrozenDictionary();
    private readonly FrozenDictionary<int, string> tokens2 = tokens.Select((token, i) => new KeyValuePair<int, string>(i, token)).ToFrozenDictionary();
    public int TokenCount => tokens.Count;

    public string GetToken(int index) => tokens2[index];

    public IEnumerable<int> Tokenize(string data)
    {
        var alternate = tokens.GetAlternateLookup<ReadOnlySpan<char>>();
        var index = 0;
        while(index < data.Length)
        {
            var span = data.AsSpan(index);
            var tokenIdx = -1;
            while(!alternate.TryGetValue(span, out tokenIdx))
            {
#if DEBUG
                if (span.Length <= 1)
                {
                    throw new InvalidOperationException($"Unknown token '{span}'");
                }
#endif
                span = span[..^1];
            }

            yield return tokenIdx;
        }
    }

    public int TokenizeSingle(string data)
    {
        if(tokens.TryGetValue(data, out var tokenIdx))
        {
            return tokenIdx;
        }

       throw new InvalidOperationException($"Unknown token '{data}'");
    }
}
