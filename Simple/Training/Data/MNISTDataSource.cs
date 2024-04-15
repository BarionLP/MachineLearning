using System.Drawing;
using System.IO.Compression;
using System.Text;

namespace Simple.Training.Data;

public sealed class MNISTDataSource {
    public MNISTDataPoint[] TrainingSet { get; }
    
    public MNISTDataSource(FileInfo mnistFileInfo) {
        using var mnistStream = mnistFileInfo.OpenRead();
        using var mnistArchive = new ZipArchive(mnistStream);

        var trainingImages = ReadImages(mnistArchive.GetEntry("train-images.idx3-ubyte")!);
        var trainingLabels = ReadLabels(mnistArchive.GetEntry("train-labels.idx1-ubyte")!);

        TrainingSet = new MNISTDataPoint[trainingImages.Length];
        foreach(var i in ..trainingImages.Length) {
            TrainingSet[i] = MNISTDataPoint.FromRaw(trainingImages[i], trainingLabels[i]);
        }
    }

    private static byte[][] ReadImages(ZipArchiveEntry entry) {
        using var stream = entry.Open();
        using var reader = new BinaryReader(stream);

        reader.ReadInt32BigEndian(); // magic starting value
        var imageCount = reader.ReadInt32BigEndian();
        var rowCount = reader.ReadInt32BigEndian();
        var columnCount = reader.ReadInt32BigEndian();

        var images = new byte[imageCount][];
        foreach(var i in ..imageCount) {
            images[i] = reader.ReadBytes(rowCount * columnCount);
        }
        return images;
    }

    private static byte[] ReadLabels(ZipArchiveEntry entry) {
        using var stream = entry.Open();
        using var reader = new BinaryReader(stream);
        
        reader.ReadInt32BigEndian(); // magic starting value
        var labelCount = reader.ReadInt32BigEndian();
        var labels = new byte[labelCount];
        foreach(var i in ..labelCount) {
            labels[i] = reader.ReadByte();
        }

        return labels;
    }
}

public sealed record MNISTDataPoint(Number[] Image, int Digit) : DataPoint<Number[], int>(Image, Digit) {
    public const int SIZE = 28; 
    public string DumpImage() {
        var sb = new StringBuilder();
        var i = 0;
        foreach(var l in ..SIZE) {
            if(l > 0) sb.AppendLine();
            foreach(var c in ..SIZE) {
                sb.Append((Image[i]*9).ToString("0"));
                sb.Append(' ');
                i++;
            }
        }

        sb.Replace('0', '.');

        return sb.ToString();
    }

    public static MNISTDataPoint FromRaw(byte[] rawImage, byte rawDigit) {
        var image = new Number[rawImage.Length];
        foreach(var i in ..rawImage.Length) {
            image[i] = rawImage[i] / 255.0;
        }

        return new(image, rawDigit);
    }
}
