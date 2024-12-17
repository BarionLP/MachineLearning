using MachineLearning.Data.Entry;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Evaluation;

public static class Evaluator
{
    public static DataSetEvaluationResult Evaluate<TInput, TOutput>(
        this EmbeddedModel<TInput, TOutput> model, 
        IEnumerable<DataEntry<TInput, TOutput>> dataSet
        ) where TInput : notnull where TOutput : notnull
    {
        int correctCounter = 0;
        int totalCounter = 0;
        foreach(var entry in dataSet)
        {
            totalCounter++;
            if(model.Process(entry.Input).Equals(entry.Expected))
            {
                correctCounter++;
            }
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = double.NaN,
        };
    }

    public static DataSetEvaluationResult Evaluate<TInput, TOutput>(
        this EmbeddedModel<TInput, TOutput> model, 
        ICostFunction costFunction, 
        IEnumerable<TrainingData<TInput, TOutput>> dataSet
        ) where TInput : notnull where TOutput : notnull
    {
        int correctCounter = 0;
        double totalCost = 0;
        int totalCounter = 0;
        foreach(var entry in dataSet)
        {
            totalCounter++;
            var outputWeights = model.InnerModel.Process(model.InputLayer.Process(entry.InputValue));
            var (output, confidence) = model.OutputLayer.Process(outputWeights);

            if(output.Equals(entry.ExpectedValue))
            {
                correctCounter++;
            }

            totalCost += costFunction.TotalCost(outputWeights, entry.Expected);
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }
}
