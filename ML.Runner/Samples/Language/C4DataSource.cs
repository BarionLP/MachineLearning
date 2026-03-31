using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Text.Json;
using System.Threading;

namespace ML.Runner.Samples.Language;

public sealed class C4DataSource : IDisposable
{
    public int CurrentFile => nextFile - 1;
    public int CurrentLinesRead => currentFile?.LinesRead ?? 0;

    private int nextFile;
    private C4FileReader? currentFile;
    private Task<FileInfo> downloadTask;
    private readonly CancellationTokenSource cts = new();

    public C4DataSource(int initalFile = 0)
    {
        nextFile = initalFile;
        downloadTask = DownloadAsync(initalFile, cts.Token);
    }

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
        downloadTask = DownloadAsync(nextFile, cts.Token);
        goto label;
    }

    public (int, int?) GetState() => (CurrentFile, CurrentLinesRead);

    public void Dispose()
    {
        currentFile?.Dispose();
        cts.Cancel();
        cts.Dispose();
    }

    public static async Task<FileInfo> DownloadAsync(int fileIndex, CancellationToken token)
    {
        var file = AssetManager.GetDataFile($"c4-train_noblock/{fileIndex:D5}-of-01024.json.gz");
        if (!file.Exists)
        {
            Console.WriteLine($"Downloading file {fileIndex:D5}...");
            using var client = new HttpClient();
            using var stream = await client.GetStreamAsync($"https://huggingface.co/datasets/allenai/c4/resolve/main/en.noblocklist/c4-train.{fileIndex:D5}-of-01024.json.gz", token);
            using var fileStream = file.Create();
            await stream.CopyToAsync(fileStream, token);
        }
        return file;
    }

    public IEnumerator<string> GetEnumerator() => GetLines().GetEnumerator();

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