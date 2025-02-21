namespace MachineLearning.Samples;

public static class AssetManager
{
    public static readonly DirectoryInfo Directory = new DirectoryInfo(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)).Directory(@"OneDrive - Schulen Stadt Schwäbisch Gmünd\Data\MachineLearning");
    public static readonly DirectoryInfo ModelDirectory = Directory.Directory("Model");
    public static readonly DirectoryInfo WeightMapsDirectory = Directory.Directory("Maps");
    public static readonly DirectoryInfo DataDirectory = Directory.Directory("Data");
    public static readonly DirectoryInfo CustomDigits = DataDirectory.Directory("Digits");
    public static readonly FileInfo MNISTArchive = GetDataFile("MNIST_ORG.zip");
    public static readonly FileInfo Sentences = GetDataFile("sentences.txt");
    public static readonly FileInfo Speech = GetDataFile("speech.txt");

    public static FileInfo GetModelFile(string fileName) => ModelDirectory.File(fileName.EndsWith(ModelSerializer.FILE_EXTENSION) ? fileName : $"{fileName}{ModelSerializer.FILE_EXTENSION}");
    public static FileInfo GetDataFile(string fileName) => DataDirectory.File(fileName);
    public static DirectoryInfo GetWeightMapFolder(string modelName) => WeightMapsDirectory.Directory(modelName);
}
