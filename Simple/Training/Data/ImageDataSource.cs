using Simple.Training.Data;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Simple;

/// <summary>
/// reads all pngs in <paramref name="directoryInfo"/> and returns them as <see cref="double[]"/>
/// </summary>
public sealed class ImageDataSource(DirectoryInfo directoryInfo){
    public MNISTDataPoint[] DataSet { get; } = directoryInfo.EnumerateFiles("*.png")
            .Select(file => new MNISTDataPoint(
                GetGrayscaleImageArray(file),
                file.NameWithoutExtension().Parse<int>()
            )).ToArray();

    public static Number[] GetGrayscaleImageArray(FileInfo imageFile){
        using Image<Rgba32> image = Image.Load<Rgba32>(imageFile.FullName);
        image.Mutate(x => x.Grayscale());

        var grayscaleValues = new Number[image.Width * image.Height];

        for (int y = 0; y < image.Height; y++){
            for (int x = 0; x < image.Width; x++){
                grayscaleValues[y * image.Width + x] = image[x, y].R / 255.0;
            }
        }

        return grayscaleValues;
    }
}
