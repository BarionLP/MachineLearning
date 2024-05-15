using MachineLearning.Data.Entry;

namespace Simple;

public sealed class SimpleSentencesDataSource
{

    public static IEnumerable<DataEntry<string, char>> GenerateData(int contextSize)
    {
        //Console.WriteLine(Data.Average(s => s.Length));
        var data = File.ReadAllText(@"sentences.txt")
            .ToLowerInvariant()
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(s => s.Length <= contextSize);
        
        foreach (var sentence in data)
        {
            for (var i = 3; i < sentence.Length; i++)
            {
                yield return new(sentence[..i], sentence[i]);
            }
        }
    }
}
