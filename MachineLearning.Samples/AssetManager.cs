namespace MachineLearning.Samples;

public static class AssetManager
{
    public static readonly DirectoryInfo Directory = new DirectoryInfo(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)).Directory(@"OneDrive - Schulen Stadt Schwäbisch Gmünd\Data\MachineLearning");
    public static readonly DirectoryInfo CustomDigits = Directory.Directory("Digits");
    public static readonly FileInfo MNISTArchive = Directory.File("MNIST_ORG.zip");
    public static readonly FileInfo Sentences = Directory.File("sentences.txt");
    public static readonly FileInfo Speech = Directory.File("speech.txt");
}
