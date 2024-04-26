namespace Simple;

public sealed class ImageInputNoise : IInputDataNoise<Number[]>{
    public required int Size { get; init; }
    public double NoiseStrength { get; init; } = 0;
    public double NoiseProbability { get; init; } = 0;
    public int MaxShift { get; init; } = 0;
    public double MaxAngle { get; init; } = 0;
    public double MinScale { get; init; } = 1;
    public double MaxScale { get; init; } = 1;
    public  Random Random { get; init; } = Random.Shared;

    public Number[] Apply(Number[] data){
        return TransformImage(data, Random.NextDouble(MinScale, MaxScale), Random.NextDouble(-MaxAngle, MaxAngle), Random.Next(-MaxShift, MaxShift), Random.Next(-MaxShift, MaxShift), Random);
    }

    private Number[] TransformImage(Number[] original, double scale, double degrees, int shiftX, int shiftY, Random random){
        var radians = degrees * Math.PI / 180;

        var transformedImage = new Number[original.Length];
        if (scale != 0){

            (double x, double y) iHat = (Math.Cos(radians) / scale, Math.Sin(radians) / scale);
            (double x, double y) jHat = (-iHat.y, iHat.x);
            for (int y = 0; y < Size; y++){
                for (int x = 0; x < Size; x++){
                    double u = x / (Size - 1.0);
                    double v = y / (Size - 1.0);

                    double uTransformed = iHat.x * (u - 0.5) + jHat.x * (v - 0.5) + 0.5 - (shiftX/(double)(Size-1));
                    double vTransformed = iHat.y * (u - 0.5) + jHat.y * (v - 0.5) + 0.5 - (shiftY/(double)(Size - 1));
                    double pixelValue = Sample(original, uTransformed, vTransformed);
                    double noiseValue = 0;
                    if (random.NextDouble() <= NoiseProbability)
                    {
                        noiseValue = (random.NextDouble() - 0.5) * 2 * NoiseStrength;
                    }
                    transformedImage[GetFlatIndex(x, y)] = Math.Clamp(pixelValue + noiseValue, 0, 1);
                }
            }
        }
        return transformedImage;
    }

    private double Sample(Number[] source, double u, double v){
        u = Math.Max(Math.Min(1, u), 0);
        v = Math.Max(Math.Min(1, v), 0);

        double texX = u * (Size - 1);
        double texY = v * (Size - 1);

        int indexLeft = (int)texX;
        int indexBottom = (int)texY;
        int indexRight = Math.Min(indexLeft + 1, Size - 1);
        int indexTop = Math.Min(indexBottom + 1, Size - 1);

        double blendX = texX - indexLeft;
        double blendY = texY - indexBottom;

        double bottomLeft = source[GetFlatIndex(indexLeft, indexBottom)];
        double bottomRight = source[GetFlatIndex(indexRight, indexBottom)];
        double topLeft = source[GetFlatIndex(indexLeft, indexTop)];
        double topRight = source[GetFlatIndex(indexRight, indexTop)];

        double valueBottom = bottomLeft + (bottomRight - bottomLeft) * blendX;
        double valueTop = topLeft + (topRight - topLeft) * blendX;
        double interpolatedValue = valueBottom + (valueTop - valueBottom) * blendY;
        return interpolatedValue;
    }

    private int GetFlatIndex(int x, int y) => x + y * Size;
}
