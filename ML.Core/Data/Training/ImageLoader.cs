using System.IO;
using StbImageSharp;

namespace ML.Core.Data.Training;

public static class ImageLoader
{
    public static double[] LoadGrayScale(FileInfo file)
    {
        using var stream = file.OpenRead();
        var image = ImageResult.FromStream(stream, ColorComponents.Grey);

        var grayscaleValues = new double[image.Data.Length];

        for (int i = 0; i < image.Data.Length; i++)
        {
            grayscaleValues[i] = image.Data[i] / 255.0;
        }

        return grayscaleValues;
    }
}