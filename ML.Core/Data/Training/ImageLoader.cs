using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ML.Core.Data.Training;

public static class ImageLoader
{
    public static double[] LoadGrayScale(FileInfo file)
    {
        using var stream = file.OpenRead();
        using var image = Image.Load<L16>(stream);
        
        var grayscaleValues = new double[image.Width * image.Height];

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                grayscaleValues[y * image.Width + x] = image[x, y].PackedValue / (double)ushort.MaxValue;
            }
        }

        return grayscaleValues;
    }
}