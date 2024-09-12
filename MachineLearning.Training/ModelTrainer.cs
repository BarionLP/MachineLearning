using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("MachineLearning.Benchmarks")]

namespace MachineLearning.Training;

public static class ModelTrainer
{
    public static LegacyModelTrainer<TInput, TOutput> Legacy<TInput, TOutput>(EmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config) where TInput : notnull where TOutput : notnull
            => new(model, config);

    public static GenericModelTrainer<TInput, TOutput> Generic<TInput, TOutput>(IGenericModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config) where TInput : notnull where TOutput : notnull
            => new(model, config);
}