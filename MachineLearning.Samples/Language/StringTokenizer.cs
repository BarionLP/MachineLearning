using MachineLearning.Data;
using System.Buffers;
using System.Collections.Frozen;

namespace MachineLearning.Samples.Language;

public sealed class StringTokenizer : ITokenizer<string>
{
    private readonly FrozenDictionary<string, int> tokens;
    private readonly FrozenDictionary<int, string> tokens2;
    private readonly string fallbackTokens;
    private readonly FrozenDictionary<string, int> altTokens;

    public int TokenCount => fallbackTokens.Length + tokens.Count;

    public StringTokenizer(HashSet<string> tokens, string fallbackTokens, IEnumerable<(string alt, string origninal)> altTokens)
    {
        this.fallbackTokens = fallbackTokens;
        this.tokens = tokens.Select((token, i) => new KeyValuePair<string, int>(token, fallbackTokens.Length + i)).ToFrozenDictionary(StringComparer.InvariantCultureIgnoreCase);
        this.tokens2 = tokens.Select((token, i) => new KeyValuePair<int, string>(fallbackTokens.Length + i, token)).ToFrozenDictionary();
        this.altTokens = altTokens.Select(t => new KeyValuePair<string, int>(t.alt, TokenizeSingle(t.origninal))).ToFrozenDictionary(StringComparer.InvariantCultureIgnoreCase);
    }

    public string GetToken(int index)
    {
        if (index < fallbackTokens.Length)
        {
            return fallbackTokens[index].ToString();
        }
        return tokens2[index];
    }

    public string Decode(IEnumerable<int> tokens) => string.Join("", tokens.Select(GetToken));

    private static readonly SearchValues<char> wordEndSymbols = SearchValues.Create(" .,:?!;");
    public IEnumerable<int> Tokenize(string data)
    {
        var alternate = tokens.GetAlternateLookup<ReadOnlySpan<char>>();
        var alternate2 = altTokens.GetAlternateLookup<ReadOnlySpan<char>>();
        var index = 0;
        while (index < data.Length)
        {
            var span = data.AsSpan(index);
            if (index == 0 || data[index - 1] == ' ')
            {
                var word = span;
                var wordLength = word.IndexOfAny(wordEndSymbols);
                if (wordLength >= 0)
                {
                    word = word[..wordLength];
                }

                if (alternate.TryGetValue(word, out var tokenIdx) || alternate2.TryGetValue(word, out tokenIdx))
                {
                    index += word.Length;
                    yield return tokenIdx;
                    continue;
                }
            }

            var fallback = fallbackTokens.IndexOf(char.ToLower(span[0]));
            if (fallback < 0)
            {
                if (!alternate2.TryGetValue(span[..1], out fallback))
                {
                    throw new InvalidOperationException($"Unknown token '{span[0]}'");
                }
            }

            index++;
            yield return fallback;
        }
    }

    public int TokenizeSingle(string data)
    {
        if (tokens.TryGetValue(data, out var tokenIdx) || altTokens?.TryGetValue(data, out tokenIdx) is true)
        {
            return tokenIdx;
        }

        if (data.Length == 1)
        {
            tokenIdx = fallbackTokens.IndexOf(data[0]);
            if (tokenIdx >= 0)
            {
                return tokenIdx;
            }
        }

        throw new InvalidOperationException($"Unknown token '{data}'");
    }
}
