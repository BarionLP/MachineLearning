using MachineLearning.Data.Entry;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MachineLearning.Data.Source;

/// <summary>
/// reads all pngs in <paramref name="directoryInfo"/> and returns them as <see cref="float[]"/>
/// </summary>
public sealed class ImageDataSource(DirectoryInfo directoryInfo)
{
    public ImageDataEntry[] DataSet { get; } = [.. directoryInfo.EnumerateFiles("*.png")
            .Select(file => new ImageDataEntry(
                GetGrayscaleImageArray(file),
                file.NameWithoutExtension.Parse<int>()
            ))];

    public static double[] GetGrayscaleImageArray(FileInfo imageFile)
    {
        using Image<Rgba32> image = Image.Load<Rgba32>(imageFile.FullName);
        image.Mutate(x => x.Grayscale());

        var grayscaleValues = new double[image.Width * image.Height];

        for(int y = 0; y < image.Height; y++)
        {
            for(int x = 0; x < image.Width; x++)
            {
                grayscaleValues[y * image.Width + x] = image[x, y].R / 255.0;
            }
        }

        return grayscaleValues;
    }
}
