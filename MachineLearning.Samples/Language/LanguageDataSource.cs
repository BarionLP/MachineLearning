using System.Text;

namespace MachineLearning.Samples.Language;

public sealed class LanguageDataSource
{
    public const string TOKENS = " !\",-.0123456789:;?_abcdefghijklmnopqrstuvwxyz?ßäöü";
    public static IEnumerable<DataEntry<string, char>> SentencesData(int contextSize)
    {
        var data = GetLines(AssetManager.Sentences.FullName)
            .Where(s => s.Length <= contextSize);

        foreach(var sentence in data)
        {
            for(var i = 3; i < sentence.Length; i++)
            {
                yield return new(sentence[..i], sentence[i]);
            }
        }
    }

    public static IEnumerable<DataEntry<string, char>> SpeechData(int contextSize)
    {
        foreach(var sentence in GetLines(AssetManager.Speech.FullName))
        {
            var start = 0;
            for(var i = 4; i < sentence.Length; i++)
            {
                if(i - start > contextSize)
                {
                    start = i - contextSize;
                }
                yield return new(sentence[start..i].Trim(), sentence[i]);
            }
        }
    }

    public static IEnumerable<string> GetLines(string path)
        => File.ReadAllText(path, Encoding.UTF8).ToLowerInvariant()
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    public static void PrepareData(string sourcePath, string targetPath, bool overrideTarget = false)
    {
        var rawData = File.ReadAllText(sourcePath, Encoding.Latin1);
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
