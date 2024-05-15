using MachineLearning.Data.Entry;
using MachineLearning.Data.Noise;
using MachineLearning.Data.Source;
using MachineLearning.Domain.Activation;
using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Training.GUI;

public class MNISTModel
{
static void Create() {
        var initer = new HeInitializer();
        var networkBuilderHuge = NetworkBuilder.Recorded<double[], int>(784)
                .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
                .SetEmbedder(MNISTEmbedder.Instance)
                .AddLayer(256, initer)
                .AddLayer(128, initer)
                .AddLayer(10, builder => builder.SetActivationMethod(SoftmaxActivation.Instance).Initialize(new XavierInitializer()));

        var networkBuilderGPT = NetworkBuilder.Recorded<double[], int>(784)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .SetEmbedder(MNISTEmbedder.Instance)
            .AddLayer(128, initer)
            .AddLayer(10, builder => builder.SetActivationMethod(SoftmaxActivation.Instance).Initialize(new XavierInitializer()));

        var mnistDataSource = new MNISTDataSource(new(@"C:\Users\Nation\OneDrive - Schulen Stadt Schwäbisch Gmünd\Data\MNIST_ORG.zip"));
        var images = new ImageDataSource(new(@"C:\Users\Nation\OneDrive\Digits"));

        var config = new TrainingConfig<double[], int>()
        {
            TrainingSet = mnistDataSource.TrainingSet,
            TestSet = mnistDataSource.TestingSet,

            EpochCount = 4,
            BatchSize = 256 * 2,

            Optimizer = new AdamOptimizerConfig
            {
                LearningRate = 0.1,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new MNISTOutputResolver(),
            InputNoise = new ImageInputNoise
            {
                Size = ImageDataEntry.SIZE,
                NoiseStrength = 0.35,
                NoiseProbability = 0.75,
                MaxShift = 2,
                MaxAngle = 30,
                MinScale = 0.8,
                MaxScale = 1.2,
            },

            DumpEvaluationAfterBatches = 4,
        };


    }
}
