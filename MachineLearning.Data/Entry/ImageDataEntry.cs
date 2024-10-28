using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MachineLearning.Data.Entry;

public sealed record ImageDataEntry(float[] Image, int Digit) : DataEntry<float[], int>(Image, Digit)
{
    public const int SIZE = 28; //TODO: un-hardcode
    public string DumpImage()
    {
        var sb = new StringBuilder();
        var i = 0;
        foreach(var l in ..SIZE)
        {
            if(l > 0)
                sb.AppendLine();
            foreach(var c in ..SIZE)
            {
                sb.Append((Image[i] * 9).ToString("0"));
                sb.Append(' ');
                i++;
            }
        }

        sb.Replace('0', '.');

        return sb.ToString();
    }

    public static ImageDataEntry FromRaw(byte[] rawImage, byte rawDigit)
    {
        var image = new float[rawImage.Length];
        foreach(var i in ..rawImage.Length)
        {
            image[i] = rawImage[i] / 255.0f;
        }

        return new(image, rawDigit);
    }

    public void SaveImage(FileInfo fileInfo)
    {
        var image = new Image<Rgba32>(SIZE, SIZE);
        foreach(var i in ..Image.Length)
        {
            int row = i / SIZE;
            int column = i % SIZE;
            image[column, row] = new Rgba32(Image[i], Image[i], Image[i]);
        }
        image.SaveAsPng(fileInfo.FullName);
    }
}
