using System.Collections.Frozen;

namespace MachineLearning.Samples.Language;

public sealed class CharOutputResolver(string tokens) : IOutputResolver<char>
{
    private readonly FrozenDictionary<char, Vector> _values = tokens.Select((c, i) =>
    {
        var vector = Vector.Create(LanguageDataSource.TOKENS.Length);
        vector[i] = 1;
        return new KeyValuePair<char, Vector>(c, vector);
    }).ToFrozenDictionary();

    public Vector Expected(char b) => _values[b];
}