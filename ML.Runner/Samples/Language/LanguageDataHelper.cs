using System;
using System.IO;
using System.Text;
using ML.Core.Data;
using ML.Core.Data.Training;

namespace ML.Runner.Samples.Language;

public static class LanguageDataHelper
{
    public static IEnumerable<TrainingEntry<int[], Vector, int>> ToTrainingData(this IEnumerable<(int[] Input, int Expected)> source, int tokenCount)
    {
        var cache = Enumerable.Range(0, tokenCount).Select(i =>
        {
            var vector = Vector.Create(tokenCount);
            vector[i] = 1;
            return new KeyValuePair<int, Vector>(i, vector);
        }).ToFrozenDictionary();

        return source.Select(MapData);

        TrainingEntry<int[], Vector, int> MapData((int[] Input, int Expected) e)
        {
            return new (e.Input, cache[e.Expected], e.Expected);
        }
    }


    public static IEnumerable<TrainingEntry<int[], Matrix, int>> ToTrainingDataMatrix(this IEnumerable<(int[] Input, int Expected)> source, int tokenCount, int contextSize, int? fillerToken)
    {
        var cache = Enumerable.Range(0, tokenCount).Select(i =>
        {
            var vector = Vector.Create(tokenCount);
            vector[i] = 1;
            return new KeyValuePair<int, Vector>(i, vector);
        }).ToFrozenDictionary();

        return source.Where(static e => e.Input.Length > 0).Select(MapData);

        TrainingEntry<int[], Matrix, int> MapData((int[] Input, int Expected) e)
        {
            return fillerToken.HasValue ? ImplFiller(fillerToken.Value) : Impl();

            TrainingEntry<int[], Matrix, int> Impl()
            {
                var length = int.Min(e.Input.Length, contextSize);
                var expected = Matrix.Create(length, tokenCount);
                var inputOffset = e.Input.Length <= contextSize ? 1 : e.Input.Length - contextSize + 1;

                Debug.Assert((length - 1) + inputOffset == e.Input.Length);
                foreach (var i in inputOffset..e.Input.Length)
                {
                    cache[e.Input[i]].CopyTo(expected.RowRef(i - inputOffset));
                }

                cache[e.Expected].CopyTo(expected.RowRef(length - 1));

                return new(fillerToken.HasValue ? e.Input.PadLeft(contextSize, fillerToken.Value) : e.Input, expected, e.Expected);
            }

            // filling with a filler in this way is probably bad but i'll use dynamic input size anyway 
            TrainingEntry<int[], Matrix, int> ImplFiller(int filler)
            {
                var length = contextSize;
                var expected = Matrix.Create(length, tokenCount);
                var inputOffset = e.Input.Length < contextSize ? 0 : e.Input.Length - contextSize + 1;

                if (e.Input.Length + 1 < contextSize)
                {
                    var embedding = cache[filler];
                    foreach (var i in ..(contextSize - e.Input.Length - 1))
                    {
                        embedding.CopyTo(expected.RowRef(i));
                    }
                }

                foreach (var i in inputOffset..e.Input.Length)
                {
                    cache[e.Input[i]].CopyTo(expected.RowRef(contextSize - e.Input.Length + i - 1));
                }

                cache[e.Expected].CopyTo(expected.RowRef(length - 1));

                return new(fillerToken.HasValue ? e.Input.PadLeft(contextSize, fillerToken.Value) : e.Input, expected, e.Expected);

            }
        }
    }

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