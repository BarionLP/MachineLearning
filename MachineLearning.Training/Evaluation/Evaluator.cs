using MachineLearning.Data.Entry;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Evaluation;

public static class Evaluator
{
    public static DataSetEvaluationResult Evaluate<TInput, TOutput>(
        this EmbeddedModel<TInput, TOutput> model,
        ICostFunction costFunction,
        IEnumerable<TrainingData<TInput, TOutput>> dataSet
        ) where TInput : notnull where TOutput : notnull
    {
        int correctCounter = 0;
        double totalCost = 0;
        int totalCounter = 0;
        foreach (var entry in dataSet)
        {
            totalCounter++;
            var outputWeights = model.InnerModel.Process(model.InputLayer.Process(entry.InputValue));
            var (output, confidence) = model.OutputLayer.Process(outputWeights);

            if (output.Equals(entry.ExpectedValue))
            {
                correctCounter++;
            }

            totalCost += costFunction.TotalCost(outputWeights, entry.ExpectedWeights);
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }

    public static DataSetEvaluationResult Evaluate<TSnapshot>(
        this IModel<Vector, TSnapshot> model,
        ICostFunction costFunction,
        IEnumerable<TrainingData<Vector, Vector>> dataSet
        ) where TSnapshot : ILayerSnapshot
    {
        double totalCost = 0;
        int totalCounter = 0;
        foreach (var entry in dataSet)
        {
            totalCounter++;
            var outputWeights = model.Process(entry.InputValue);

            totalCost += costFunction.TotalCost(outputWeights, entry.ExpectedWeights);
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = 0,
            TotalCost = totalCost,
        };
    }
}
