using MachineLearning.Samples.MNIST;
using MachineLearning.Training;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization.Adam;
using ML.MultiLayerPerceptron;

namespace ML.Tests;

public sealed class MNISTTest
{
    [Test]
    public async Task TrainMNIST()
    {
        var random = new Random(69);

        var model = MNISTModel.CreateModel(random);
        DataSetEvaluation? evaluation = null;

        var config = new TrainingConfig()
        {
            EpochCount = 1,
            Threading = ThreadingMode.Full,
            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.0046225016f,
                CostFunction = CrossEntropyLoss.Instance,
            },

            DumpEvaluationAfterBatches = 1,
            EvaluationCallback = data =>
            {
                evaluation = data;
                //Console.WriteLine(data.Dump());
            },
            RandomSource = random,
        };
        var trainingSet = MNISTModel.GetTrainingSet(random);

        await Assert.That(model.InnerModel.WeightCount).IsEqualTo(235146);

        var trainer = new EmbeddedModelTrainer<double[], int>(model, config, trainingSet);
        trainer.Train();

        await Assert.That(evaluation).IsNotNull();
        await Assert.That(evaluation!.Result.AverageCost).IsBetween(0.7, 0.8);
    }
}
