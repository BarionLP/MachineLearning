using System.IO.Compression;
using MachineLearning.Data.Entry;

namespace MachineLearning.Data.Source;

public sealed class MNISTDataSource
{
    public ImageDataEntry[] TrainingSet { get; }
    public ImageDataEntry[] TestingSet { get; }

    public MNISTDataSource(FileInfo mnistFileInfo)
    {
        using var mnistStream = mnistFileInfo.OpenRead();
        using var mnistArchive = new ZipArchive(mnistStream);

        var trainingImages = ReadImages(mnistArchive.GetEntry("train-images.idx3-ubyte")!);
        var trainingLabels = ReadLabels(mnistArchive.GetEntry("train-labels.idx1-ubyte")!);

        TrainingSet = new ImageDataEntry[trainingImages.Length];
        foreach (var i in ..trainingImages.Length)
        {
            TrainingSet[i] = ImageDataEntry.FromRaw(trainingImages[i], trainingLabels[i]);
        }

        var testingImages = ReadImages(mnistArchive.GetEntry("t10k-images.idx3-ubyte")!);
        var testingLabels = ReadLabels(mnistArchive.GetEntry("t10k-labels.idx1-ubyte")!);

        TestingSet = new ImageDataEntry[testingImages.Length];
        foreach (var i in ..testingImages.Length)
        {
            TestingSet[i] = ImageDataEntry.FromRaw(testingImages[i], testingLabels[i]);
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
