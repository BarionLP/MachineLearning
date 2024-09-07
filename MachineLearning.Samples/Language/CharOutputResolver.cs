using System.Collections.Frozen;

namespace MachineLearning.Samples.Language;

public sealed class CharOutputResolver(string tokens) : IOutputResolver<char>
{
    private readonly FrozenDictionary<char, Vector> _values = tokens.Select((c, i) =>
    {
        var vector = Vector.Create(tokens.Length);
        vector[i] = 1;
        return new KeyValuePair<char, Vector>(c, vector);
    }).ToFrozenDictionary();

    public Vector Expected(char b)
    {
        if(_values.TryGetValue(b, out var value))
        {
            return value;
        }

        throw new ArgumentException($"Unknown Token '{b}'");
    }
}