namespace ML.Core.Training;

public sealed class AdamOptimizer : Optimizer
{
    public static ModuleOptimizerRegistry<AdamOptimizer> Registry { get; } = [];
    protected override ModuleOptimizerRegistry RegistryGetter => Registry;
    public Weight FirstDecayRate { get; init; } = 0.9f;
    public Weight SecondDecayRate { get; init; } = 0.99f; // or 0.999f
    public Weight Epsilon { get; init; } = 1e-8f;

    public Weight Iteration
    {
        get;
        set
        {
            field = value;
            CurrentFirstCorrection = 1 - Weight.Pow(FirstDecayRate, Iteration);
            CurrentSecondCorrection = 1 - Weight.Pow(SecondDecayRate, Iteration);
            WeightReductionOperation = new(this);
        }
    }

    public AdamFirstMomentEstimateOperation FirstMomentEstimateOperation { get; private set; }
    public AdamSecondMomentEstimateOperation SecondMomentEstimateOperation { get; private set; }
    public AdamWeightReductionOperation WeightReductionOperation { get; private set; }

    public Weight CurrentFirstCorrection { get; private set; }
    public Weight CurrentSecondCorrection { get; private set; }


    public override void Init()
    {
        Iteration = 1;  // even when retraining!
        FirstMomentEstimateOperation = new(FirstDecayRate);
        SecondMomentEstimateOperation = new(SecondDecayRate);
    }

    public override void OnBatchCompleted()
    {
        Iteration++;
    }

    public readonly struct AdamFirstMomentEstimateOperation(Weight decayRate) : IBinaryOperator<AdamFirstMomentEstimateOperation>
    {
        private readonly Weight decayRate = decayRate;

        public static Weight Invoke(in AdamFirstMomentEstimateOperation state, Weight lastMoment, Weight gradient) => state.decayRate * lastMoment + (1 - state.decayRate) * gradient;
        public static SimdVector Invoke(in AdamFirstMomentEstimateOperation state, SimdVector lastMoment, SimdVector gradient) => state.decayRate * lastMoment + (1 - state.decayRate) * gradient;
    }

    public readonly struct AdamSecondMomentEstimateOperation(Weight decayRate) : IBinaryOperator<AdamSecondMomentEstimateOperation>
    {
        private readonly Weight decayRate = decayRate;
        public static Weight Invoke(in AdamSecondMomentEstimateOperation state, Weight lastMoment, Weight gradient) => state.decayRate * lastMoment + (1 - state.decayRate) * gradient * gradient;
        public static SimdVector Invoke(in AdamSecondMomentEstimateOperation state, SimdVector lastMoment, SimdVector gradient) => state.decayRate * lastMoment + (1 - state.decayRate) * gradient * gradient;
    }

    public readonly struct AdamWeightReductionOperation(AdamOptimizer context) : IBinaryOperator<AdamWeightReductionOperation>
    {
        private readonly Weight learningRate = context.LearningRate;
        private readonly Weight firstMomentCorrection = context.CurrentFirstCorrection;
        private readonly Weight secondMomentCorrection = context.CurrentSecondCorrection;
        private readonly Weight epsilon = context.Epsilon;

        public static Weight Invoke(in AdamWeightReductionOperation state, Weight firstMoment, Weight secondMoment)
        {
            var mHat = firstMoment / state.firstMomentCorrection;
            var vHat = secondMoment / state.secondMomentCorrection;
            return state.learningRate * mHat / (Weight.Sqrt(vHat) + state.epsilon);
        }
        public static SimdVector Invoke(in AdamWeightReductionOperation state, SimdVector firstMoment, SimdVector secondMoment)
        {
            var mHat = firstMoment / state.firstMomentCorrection;
            var vHat = secondMoment / state.secondMomentCorrection;
            return state.learningRate * mHat / (SimdVectorHelper.SquareRoot(vHat) + SimdVectorHelper.Create(state.epsilon));
        }
    }
}
