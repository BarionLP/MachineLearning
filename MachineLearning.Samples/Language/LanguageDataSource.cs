using System.Text;

namespace MachineLearning.Samples.Language;

public static class LanguageDataSource
{
    public const string TOKENS = " !\",-.0123456789:;?_abcdefghijklmnopqrstuvwxyzßäöü"; // add: %
    public static IEnumerable<DataEntry<string, char>> SentencesData(int contextSize) 
        => GetLines(AssetManager.Sentences.FullName).InContextSize(contextSize).ExpandPerChar();

    public static IEnumerable<DataEntry<string, char>> SpeechData(int contextSize) 
        => GetLines(AssetManager.Speech.FullName).SlidingWindow(contextSize);

    public static IEnumerable<string> InContextSize(this IEnumerable<string> data, int contextSize) 
        => data.Where(s => s.Length <= contextSize);

    public static IEnumerable<DataEntry<string, char>> ExpandPerChar(this IEnumerable<string> data)
    {
        foreach (var sentence in data)
        {
            for (var i = 3; i < sentence.Length; i++)
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
                yield return new (sentence[start..i], sentence[i]);
            }
        }
    }

    public static IEnumerable<char> GetInvalidChars(IEnumerable<string> lines) => lines.SelectMany(l => l.Where(c => !TOKENS.Contains(c)));

    public static IEnumerable<string> GetLines(FileInfo fileInfo) => GetLines(fileInfo.FullName);
    public static IEnumerable<string> GetLines(string path)
        => File.ReadAllText(path, Encoding.UTF8).ToLowerInvariant()
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    public static void PrepareData(string sourcePath, string targetPath, bool overrideTarget = false)
    {
        var rawData = File.ReadAllText(sourcePath, Encoding.UTF8);
        var sentences = ParseSentences();

        if(overrideTarget)
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
            foreach(var i in ..rawData.Length)
            {
                switch(rawData[i])
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
