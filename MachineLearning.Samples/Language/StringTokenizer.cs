using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Frozen;
using System.Diagnostics;
using MachineLearning.Data;

namespace MachineLearning.Samples.Language;

public sealed class StringTokenizer : ITokenizer<string>
{
    private readonly FrozenDictionary<string, int> textToToken;
    private readonly FrozenDictionary<int, string> tokenToText;
    private readonly string fallbackTokens;
    private readonly FrozenDictionary<string, int> altTokens;

    public int TokenCount => fallbackTokens.Length + textToToken.Count;

    public StringTokenizer(HashSet<string> tokens, string fallbackTokens, IEnumerable<(string alt, string original)> altTokens)
    {
        this.fallbackTokens = fallbackTokens;
        this.textToToken = tokens.Select((token, i) => new KeyValuePair<string, int>(token, fallbackTokens.Length + i)).ToFrozenDictionary(StringComparer.InvariantCultureIgnoreCase);
        this.tokenToText = tokens.Select((token, i) => new KeyValuePair<int, string>(fallbackTokens.Length + i, token)).ToFrozenDictionary();
        this.altTokens = altTokens.Select(t => new KeyValuePair<string, int>(t.alt, TokenizeSingle(t.original))).ToFrozenDictionary(StringComparer.InvariantCultureIgnoreCase);

        Debug.Assert(textToToken.Count == tokenToText.Count);
        Debug.Assert(this.altTokens.All(p => tokenToText.ContainsKey(p.Value) || p.Value < fallbackTokens.Length));
    }

    public string GetToken(int index)
    {
        if (index < fallbackTokens.Length)
        {
            return fallbackTokens[index].ToString();
        }
        return tokenToText[index];
    }

    public string Decode(IEnumerable<int> tokens) => string.Join("", tokens.Select(GetToken));

    private static readonly SearchValues<char> wordEndSymbols = SearchValues.Create(" .,:?!;");
    public static ConcurrentDictionary<char, int> UnknownTokens { get; } = new();
    public IEnumerable<int> Tokenize(string data)
    {
        var tokenLookup = textToToken.GetAlternateLookup<ReadOnlySpan<char>>();
        var altTokenLookup = altTokens.GetAlternateLookup<ReadOnlySpan<char>>();
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

                if (tokenLookup.TryGetValue(word, out var tokenIdx) || altTokenLookup.TryGetValue(word, out tokenIdx))
                {
                    index += word.Length;
                    yield return tokenIdx;
                    continue;
                }
            }

            var fallback = fallbackTokens.IndexOf(char.ToLower(span[0]));
            if (fallback < 0)
            {
                if (!altTokenLookup.TryGetValue(span[..1], out fallback))
                {
                    UnknownTokens.AddOrUpdate(span[0], 1, static (_, v) => v + 1);
                    throw new InvalidOperationException($"Unknown token '{span[0]}'");
                }
            }

            index++;
            yield return fallback;
        }
    }

    public int TokenizeSingle(string data)
    {
        if (textToToken.TryGetValue(data, out var tokenIdx) || altTokens?.TryGetValue(data, out tokenIdx) is true)
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
