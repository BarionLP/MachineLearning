using System.Globalization;
using System.Windows;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Samples;
using MachineLearning.Samples.MNIST;
using MachineLearning.Serialization;
using MachineLearning.Serialization.Activation;
using SkiaSharp;

namespace MachineLearning.Training.GUI;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public TrainingProgressTracker ProgressTracker { get; } = new();


    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;

        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        ActivationMethodSerializer.RegisterDefaults();

        var serializer = new ModelSerializer(AssetManager.GetModelFile("mnist.nnw"));
        //var model = MNISTModel.GetModel();
        var model = serializer.Load(MNISTEmbedder.Instance).ReduceOrThrow();
        var config = MNISTModel.GetTrainingConfig();

        var trainer = ProgressTracker.CreateLinkedTrainer("Binary Classifier", SKColors.Blue, model, config);

        Loaded += async (sender, args) =>
        {
            StatusLabel.Content = "Training...";
            await Task.Run(trainer.Train);
            serializer.Save(model);
            StatusLabel.Content = "Done!";
            //Task.Run(trainerTiny.Train);
        };
    }
}