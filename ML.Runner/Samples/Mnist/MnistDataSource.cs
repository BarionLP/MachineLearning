using System.IO;
using System.IO.Compression;
using System.Text;
using ML.Core.Data.Training;
using ML.Core.Data.Noise;

namespace ML.Runner.Samples.Mnist;

public sealed class MnistImageSource(IEnumerable<MnistImage> images) : ITrainingDataSource<TrainingEntry<double[], Vector, int>>
{
    public bool ShuffleOnReset { get; init; } = true;
    public Random Random { get; init; } = Random.Shared;
    public required int BatchCount { get; init; }
    public IDataNoise<double[]> Noise { get; init; } = NoDataNoise<double[]>.Instance;
    public int BatchSize => data.Length / BatchCount;

    private readonly MnistImage[] data = [.. images];

    public IEnumerable<IEnumerable<TrainingEntry<double[], Vector, int>>> GetBatches()
    {
        var batchSize = BatchSize;
        foreach (var i in ..BatchCount)
        {
            yield return Batch.Create(data, i * batchSize, batchSize).Select(d => new TrainingEntry<double[], Vector, int>(Noise.Apply(d.Image), Expected(d.Digit), d.Digit));
        }
    }

    public void Reset()
    {
        if (ShuffleOnReset)
        {
            Random.Shuffle(data);
        }
    }

    private static readonly FrozenDictionary<int, Vector> _map = new Dictionary<int, Vector>() {
            { 0, Vector.Of([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
            { 1, Vector.Of([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])},
            { 2, Vector.Of([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])},
            { 3, Vector.Of([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])},
            { 4, Vector.Of([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])},
            { 5, Vector.Of([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])},
            { 6, Vector.Of([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])},
            { 7, Vector.Of([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])},
            { 8, Vector.Of([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])},
            { 9, Vector.Of([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])},
        }.ToFrozenDictionary();

    private Vector Expected(int output) => _map[output];
}

public sealed class MnistDataSet
{
    public MnistImage[] TrainingSet { get; }
    public MnistImage[] TestingSet { get; }

    public MnistDataSet(FileInfo mnistFileInfo)
    {
        using var mnistStream = mnistFileInfo.OpenRead();
        using var mnistArchive = new ZipArchive(mnistStream);

        var trainingImages = ReadImages(mnistArchive.GetEntry("train-images.idx3-ubyte")!);
        var trainingLabels = ReadLabels(mnistArchive.GetEntry("train-labels.idx1-ubyte")!);

        TrainingSet = new MnistImage[trainingImages.Length];
        foreach (var i in ..trainingImages.Length)
        {
            TrainingSet[i] = MnistImage.FromRaw(trainingImages[i], trainingLabels[i]);
        }

        var testingImages = ReadImages(mnistArchive.GetEntry("t10k-images.idx3-ubyte")!);
        var testingLabels = ReadLabels(mnistArchive.GetEntry("t10k-labels.idx1-ubyte")!);

        TestingSet = new MnistImage[testingImages.Length];
        foreach (var i in ..testingImages.Length)
        {
            TestingSet[i] = MnistImage.FromRaw(testingImages[i], testingLabels[i]);
        }
    }

    private static byte[][] ReadImages(ZipArchiveEntry entry)
    {
        using var stream = entry.Open();
        using var reader = new BinaryReader(stream);

        reader.ReadInt32BigEndian(); // magic starting value
        var imageCount = reader.ReadInt32BigEndian();
        var rowCount = reader.ReadInt32BigEndian();
        var columnCount = reader.ReadInt32BigEndian();

        var images = new byte[imageCount][];
        foreach (var i in ..imageCount)
        {
            images[i] = reader.ReadBytes(rowCount * columnCount);
        }
        return images;
    }

    private static byte[] ReadLabels(ZipArchiveEntry entry)
    {
        using var stream = entry.Open();
        using var reader = new BinaryReader(stream);

        reader.ReadInt32BigEndian(); // magic starting value
        var labelCount = reader.ReadInt32BigEndian();
        var labels = new byte[labelCount];
        foreach (var i in ..labelCount)
        {
            labels[i] = reader.ReadByte();
        }

        return labels;
    }
}

public sealed record MnistImage(double[] Image, int Digit)
{
    public const int SIZE = 28; // TODO: un-hardcode
    public string DumpImage()
    {
        var sb = new StringBuilder();
        var i = 0;
        foreach (var l in ..SIZE)
        {
            if (l > 0)
                sb.AppendLine();
            foreach (var c in ..SIZE)
            {
                sb.Append((Image[i] * 9).ToString("0"));
                sb.Append(' ');
                i++;
            }
        }

        sb.Replace('0', '.');

        return sb.ToString();
    }

    public static MnistImage FromRaw(byte[] rawImage, byte rawDigit)
    {
        var image = new double[rawImage.Length];
        foreach (var i in ..rawImage.Length)
        {
            image[i] = rawImage[i] / 255.0;
        }

        return new(image, rawDigit);
    }

    // public void SaveImage(FileInfo fileInfo)
    // {
    //     var image = new Image<Rgba32>(SIZE, SIZE);
    //     foreach (var i in ..Image.Length)
    //     {
    //         int row = i / SIZE;
    //         int column = i % SIZE;
    //         image[column, row] = new Rgba32((float)Image[i], (float)Image[i], (float)Image[i]);
    //     }
    //     image.SaveAsPng(fileInfo.FullName);
    // }
}
