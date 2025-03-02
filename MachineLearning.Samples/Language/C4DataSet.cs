using System.IO.Compression;
using System.Text.Json;
using MachineLearning.Data;

namespace MachineLearning.Samples.Language;

public sealed class C4DataSet(ITokenizer<string> tokenizer, int contextSize, int initalFile = 0) : ITrainingSet, IDisposable
{
    public int BatchCount { get; } = int.MaxValue;
    public required int BatchSize { get; init; }
    public int CurrentFile => nextFile - 1;
    public int CurrentLinesRead => currentFile?.LinesRead ?? 0;
    private int nextFile = initalFile;
    private C4FileReader? currentFile;

    private readonly ITokenizer<string> tokenizer = tokenizer;
    private readonly int contextSize = contextSize;
    private Task<FileInfo> downloadTask = Download(initalFile);


    public IEnumerable<Batch> GetBatches()
    {
        while (true)
        {
            yield return new Batch(GetTrainingData().Take(BatchSize));
        }
    }

    private IEnumerator<TrainingData>? dataEnumerator;
    public IEnumerable<TrainingData> GetTrainingData()
    {
        while (true)
        {
            while (dataEnumerator is null)
            {
                try
                {
                    dataEnumerator = tokenizer.Tokenize(NextLine()).ToArray().SlidingWindow(tokenizer.TokenizeSingle("\0"), contextSize).ToTrainingDataMatrix(tokenizer.TokenCount, contextSize, tokenizer.TokenizeSingle("\0")).GetEnumerator();
                }
                catch (Exception) { /* Console.WriteLine(e.Message); */ }
            }

            while (dataEnumerator.MoveNext())
            {
                yield return dataEnumerator.Current;
            }
            dataEnumerator = null;
        }
    }



    public IEnumerable<int[]> GetTokenizedLines()
        => GetLines().TokenizeSkipInvalid(tokenizer);

    public IEnumerable<string> GetLines()
    {
        while (true)
        {
            string? line;
            try
            {
                line = NextLine();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                yield break;
            }
            yield return line;
        }
    }

    private string NextLine()
    {
    label:
        if (currentFile?.ReadLine() is string line)
        {
            return line;
        }

        currentFile?.Dispose();
        if (!downloadTask.IsCompleted)
        {
            Console.WriteLine("Waiting for download to complete...");
        }
        if (downloadTask.IsFaulted)
        {
            Console.WriteLine($"Download failed: {downloadTask.Exception?.Message}");
        }
        Task.WaitAll(downloadTask);
        currentFile = new C4FileReader(downloadTask.Result);
        nextFile++;
        downloadTask = Download(nextFile);
        goto label;
    }

    public void Dispose()
    {
        currentFile?.Dispose();
    }

    public static async Task<FileInfo> Download(int fileIndex)
    {
        var file = AssetManager.GetDataFile($"c4-train_noblock/{fileIndex:D5}-of-01024.json.gz");
        if (!file.Exists)
        {
            Console.WriteLine($"Downloading file {fileIndex:D5}...");
            using var client = new HttpClient();
            using var stream = await client.GetStreamAsync($"https://huggingface.co/datasets/allenai/c4/resolve/main/en.noblocklist/c4-train.{fileIndex:D5}-of-01024.json.gz");
            using var fileStream = file.Create();
            await stream.CopyToAsync(fileStream);
        }
        return file;
    }

    private sealed class C4FileReader : IDisposable
    {
        private readonly Stream stream;
        private readonly GZipStream gzStream;
        private readonly StreamReader reader;
        public int LinesRead { get; private set; }

        public C4FileReader(FileInfo fileInfo)
        {
            stream = fileInfo.OpenRead();
            gzStream = new GZipStream(stream, CompressionMode.Decompress);
            reader = new StreamReader(gzStream);
        }

        public string? ReadLine()
        {
        start:
            if (reader.ReadLine() is string line)
            {
                var doc = JsonDocument.Parse(line);
                if (doc.RootElement.TryGetProperty("text", out var text) && text.GetString() is string textString)
                {
                    LinesRead++;
                    return textString;
                }
                goto start;
            }
            return null;
        }

        public void Dispose()
        {
            reader.Dispose();
            gzStream.Dispose();
            stream.Dispose();
        }
    }
}