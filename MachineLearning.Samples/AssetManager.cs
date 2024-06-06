namespace MachineLearning.Samples;

public static class AssetManager
{
    public static readonly DirectoryInfo Directory = new DirectoryInfo(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)).Directory(@"OneDrive - Schulen Stadt Schwäbisch Gmünd\Data\MachineLearning");
    public static readonly DirectoryInfo ModelDirectory = Directory.Directory("Model");
    public static readonly DirectoryInfo DataDirectory = Directory.Directory("Data");
    public static readonly DirectoryInfo CustomDigits = DataDirectory.Directory("Digits");
    public static readonly FileInfo MNISTArchive = Directory.File("MNIST_ORG.zip");
    public static readonly FileInfo Sentences = GetDataFile("sentences.txt");
    public static readonly FileInfo Speech = GetDataFile("speech.txt");

    public static FileInfo GetModelFile(string fileName) => ModelDirectory.File(fileName);
    public static FileInfo GetDataFile(string fileName) => DataDirectory.File(fileName);
}
