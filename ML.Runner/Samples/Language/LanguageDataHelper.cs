using System;
using System.IO;
using System.Text;
using ML.Core.Data;
using ML.Core.Data.Training;

namespace ML.Runner.Samples.Language;

public static class LanguageDataHelper
{
    public static IEnumerable<TrainingEntry<int[], Vector, int>> ToTrainingData(this IEnumerable<(int[], int)> source, int tokenCount)
    {
        var cache = BuildExpectedWeightCache(tokenCount);

        return source.Select(MapData);

        TrainingEntry<int[], Vector, int> MapData((int[] Input, int Expected) e)
        {
            return new(e.Input, cache[e.Expected], e.Expected);
        }
    }


    public static IEnumerable<TrainingEntry<int[], Matrix, int>> ToTrainingDataMatrix(this IEnumerable<(int[] Input, int Expected)> source, int tokenCount)
    {
        var cache = BuildExpectedWeightCache(tokenCount);

        return source.Where(static e => e.Input.Length > 0).Select(p => ToTrainingDataMatrix(p.Input, p.Expected, cache, tokenCount));
    }

    public static TrainingEntry<int[], Matrix, int> ToTrainingDataMatrix(int[] input, int expected, FrozenDictionary<int, Vector> cache, int tokenCount)
    {
        var expectedWeights = Matrix.Create(input.Length, tokenCount);

        foreach (var i in 1..input.Length)
        {
            cache[input[i]].CopyTo(expectedWeights.RowRef(i - 1));
        }

        cache[expected].CopyTo(expectedWeights.RowRef(input.Length - 1));

        return new(input, expectedWeights, expected);
    }

    public static FrozenDictionary<int, Vector> BuildExpectedWeightCache(int tokenCount) => Enumerable.Range(0, tokenCount).Select(i =>
    {
        var vector = Vector.Create(tokenCount);
        vector[i] = 1;
        return KeyValuePair.Create(i, vector);
    }).ToFrozenDictionary();

    public static IEnumerable<IEnumerable<int>> Tokenize(this IEnumerable<string> source, ITokenizer<string> tokenizer)
        => source.Select(tokenizer.Tokenize);

    // private static readonly HashSet<char> KnownInvalidTokens = [];
    public static IEnumerable<int[]> TokenizeSkipInvalid(this IEnumerable<string> source, ITokenizer<string> tokenizer)
    {
        foreach (var sentence in source)
        {
            int[] tokens;
            try
            {
                tokens = [.. tokenizer.Tokenize(sentence)];
            }
            catch (Exception)
            {
                // if(KnownInvalidTokens.Add(e.Message[15]))
                // {
                //     Console.WriteLine(e.Message);
                // }
                continue;
            }
            yield return tokens;
        }
    }

    public static IEnumerable<string> InContextSize(this IEnumerable<string> data, int contextSize)
        => data.Where(s => s.Length <= contextSize);

    public static IEnumerable<(int[] Input, int Expected)> ExpandPerToken(this IEnumerable<IEnumerable<int>> data, int endToken, int contextSize)
        => data.Select(Enumerable.ToArray).ExpandPerToken(endToken, contextSize);

    public static IEnumerable<(int[] Input, int Expected)> ExpandPerToken(this IEnumerable<int[]> data, int endToken, int contextSize)
    {
        foreach (var sentence in data)
        {
            var max = int.Min(contextSize, sentence.Length);
            for (var i = 0; i < max; i++)
            {
                yield return (sentence[..i], sentence[i]);
            }
            if (sentence.Length <= contextSize && sentence[^1] != endToken)
            {
                yield return (sentence, endToken);
            }
        }
    }

    public static IEnumerable<(int[] Input, int Expected)> SlidingWindow(this IEnumerable<IEnumerable<int>> data, int? endToken, int contextSize)
        => data.Select(Enumerable.ToArray).SlidingWindow(endToken, contextSize);
    public static IEnumerable<(int[] Input, int Expected)> SlidingWindow(this IEnumerable<int[]> data, int? endToken, int contextSize)
        => data.SelectMany(d => d.SlidingWindow(endToken, contextSize));
    public static IEnumerable<(int[] Input, int Expected)> SlidingWindow(this int[] data, int? endToken, int contextSize)
    {
        var start = 0;
        for (var i = 0; i < data.Length; i++)
        {
            yield return (data[start..i], data[i]);
            if (i - start >= contextSize)
            {
                start++;
            }
        }

        if (endToken.HasValue && data[^1] != endToken.Value)
        {
            yield return (data[^Math.Min(contextSize, data.Length)..], endToken.Value);
        }
    }

    public static IEnumerable<string> GetLines(FileInfo fileInfo) => GetLines(fileInfo.FullName);
    public static IEnumerable<string> GetLines(string path)
        => File.ReadAllText(path, Encoding.UTF8).ToLowerInvariant()
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    //.Select(line => line.Replace('-', TOKENS[4]));

    public static void PrepareData(string sourcePath, string targetPath, bool overrideTarget = false)
    {
        var rawData = File.ReadAllText(sourcePath, Encoding.UTF8);
        var sentences = ParseSentences();

        if (overrideTarget)
        {
            File.Delete(targetPath);
            File.WriteAllLines(targetPath, sentences, Encoding.UTF8);
        }
        else
        {
            File.AppendAllLines(targetPath, sentences, Encoding.UTF8);
        }

        IEnumerable<string> ParseSentences()
        {
            var start = 0;
            foreach (var i in ..rawData.Length)
            {
                switch (rawData[i])
                {
                    case '.':
                    case '?':
                    case '!':
                    //case ':':
                    case ';':
                        yield return rawData.AsSpan()[start..(i + 1)].Trim().ToString();
                        start = i + 1;
                        break;
                }
            }
        }
    }
}