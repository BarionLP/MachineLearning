using System.Collections.Frozen;
using System.Diagnostics;
using System.Text;
using MachineLearning.Data;

namespace MachineLearning.Samples.Language;

public static class LanguageDataHelper
{
    public static IEnumerable<TrainingData> ToTrainingData(this IEnumerable<DataEntry<string, char>> source, ITokenizer<string> tokenizer)
    {
        var cache = Enumerable.Range(0, tokenizer.TokenCount).Select(i =>
        {
            var vector = Vector.Create(tokenizer.TokenCount);
            vector[i] = 1;
            return new KeyValuePair<int, Vector>(i, vector);
        }).ToFrozenDictionary();

        return source.Select(MapData);


        TrainingData MapData(DataEntry<string, char> e)
        {
            var input = tokenizer.Tokenize(e.Input).ToArray();
            var expectedToken = tokenizer.TokenizeSingle(e.Expected.ToString());

            return new TrainingData<int[], int>(input, expectedToken, cache[expectedToken]);
        }
    }

    public static IEnumerable<TrainingData> ToTrainingData(this IEnumerable<DataEntry<int[], int>> source, int tokenCount)
    {
        var cache = Enumerable.Range(0, tokenCount).Select(i =>
        {
            var vector = Vector.Create(tokenCount);
            vector[i] = 1;
            return new KeyValuePair<int, Vector>(i, vector);
        }).ToFrozenDictionary();

        return source.Select(MapData);

        TrainingData MapData(DataEntry<int[], int> e)
        {
            return new TrainingData<int[], int>(e.Input, e.Expected, cache[e.Expected]);
        }
    }


    public static IEnumerable<TrainingData> ToTrainingDataMatrix(this IEnumerable<(int[] Input, int Expected)> source, int tokenCount, int contextSize, int? fillerToken)
    {
        var cache = Enumerable.Range(0, tokenCount).Select(i =>
        {
            var vector = Vector.Create(tokenCount);
            vector[i] = 1;
            return new KeyValuePair<int, Vector>(i, vector);
        }).ToFrozenDictionary();

        return source.Where(e => e.Input.Length > 0).Select(MapData);

        TrainingData MapData((int[] Input, int Expected) e)
        {
            var expected = Matrix.Create(contextSize, tokenCount);
            var inputStartIndex = Math.Max(0, e.Input.Length - contextSize + 1);

            if (fillerToken.HasValue && inputStartIndex == 0)
            {
                var embedding = cache[fillerToken.Value];
                foreach (var i in ..(contextSize - e.Input.Length - 1))
                {
                    embedding.CopyTo(expected.RowRef(i));
                }
            }


            foreach (var i in inputStartIndex..e.Input.Length)
            {
                cache[e.Input[i]].CopyTo(expected.RowRef(contextSize - e.Input.Length + i - 1));
            }

            cache[e.Expected].CopyTo(expected.RowRef(contextSize - 1));

            return new TrainingData<int[], int>(fillerToken.HasValue ? e.Input.PadLeft(contextSize, fillerToken.Value) : e.Input, e.Expected, expected.Storage);
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
                tokens = tokenizer.Tokenize(sentence).ToArray();
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

    public static IEnumerable<DataEntry<string, char>> SentencesData(int contextSize)
        => GetLines(AssetManager.Sentences.FullName).InContextSize(contextSize).ExpandPerChar();

    public static IEnumerable<DataEntry<string, char>> SpeechData(int contextSize)
        => GetLines(AssetManager.Speech.FullName).SlidingWindow(contextSize);

    public static IEnumerable<string> InContextSize(this IEnumerable<string> data, int contextSize)
        => data.Where(s => s.Length <= contextSize);

    public static IEnumerable<DataEntry<int[], int>> ExpandPerToken(this IEnumerable<IEnumerable<int>> data, int endToken, int contextSize)
        => data.Select(Enumerable.ToArray).ExpandPerToken(endToken, contextSize);

    public static IEnumerable<DataEntry<int[], int>> ExpandPerToken(this IEnumerable<int[]> data, int endToken, int contextSize)
    {
        foreach (var sentence in data)
        {
            var max = int.Min(contextSize, sentence.Length);
            for (var i = 0; i < max; i++)
            {
                yield return new(sentence[..i], sentence[i]);
            }
            if (sentence.Length <= contextSize && sentence[^1] != endToken)
            {
                yield return new(sentence, endToken);
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

    public static IEnumerable<DataEntry<string, char>> ExpandPerChar(this IEnumerable<string> data)
    {
        foreach (var sentence in data)
        {
            for (var i = 0; i < sentence.Length; i++)
            {
                yield return new(sentence[..i], sentence[i]);
            }
        }
    }

    public static IEnumerable<DataEntry<string, char>> SlidingWindow(this IEnumerable<string> data, int contextSize)
    {
        foreach (var sentence in data)
        {
            var start = 0;
            for (var i = 4; i < sentence.Length; i++)
            {
                if (i - start > contextSize)
                {
                    start = sentence.AsSpan()[start..].IndexOf(' ') + 1 + start;
                }
                yield return new(sentence[start..i], sentence[i]);
            }
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
