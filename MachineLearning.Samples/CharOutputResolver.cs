using System.Collections.Frozen;

namespace MachineLearning.Samples;

public sealed class CharOutputResolver : IOutputResolver<char> {
    private static readonly FrozenDictionary<char, Vector> _values = LanguageDataSource.TOKENS.Select((c, i) => {
        var vector = Vector.Create(LanguageDataSource.TOKENS.Length);
        vector[i] = 1;
        return new KeyValuePair<char, Vector>(c, vector);
    }).ToFrozenDictionary();

    public Vector Expected(char b) => _values[b];
}