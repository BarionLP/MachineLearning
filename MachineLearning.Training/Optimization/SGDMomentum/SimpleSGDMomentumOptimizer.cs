// using MachineLearning.Model.Layer;
// using MachineLearning.Model.Layer.Snapshot;
// using MachineLearning.Training.Cost;

// namespace MachineLearning.Training.Optimization.SGDMomentum;

// public sealed class SimpleSGDMomentumOptimizer : ILayerOptimizer<FeedForwardLayer, LayerSnapshots.Simple>
// {
//     public FeedForwardLayer Layer { get; }
//     public readonly Matrix CostGradientWeights;
//     public readonly Vector CostGradientBiases;
//     public readonly Matrix WeightVelocities;
//     public readonly Vector BiasVelocities;
//     public ICostFunction CostFunction => Optimizer.CostFunction;
//     public SGDMomentumOptimizer Optimizer { get; }

//     public SimpleSGDMomentumOptimizer(SGDMomentumOptimizer optimizer, FeedForwardLayer layer)
//     {
//         Optimizer = optimizer;
//         Layer = layer;
//         CostGradientWeights = Matrix.Create(Layer.OutputNodeCount, Layer.InputNodeCount);
//         CostGradientBiases = Vector.Create(Layer.OutputNodeCount);
//         WeightVelocities = Matrix.Create(Layer.OutputNodeCount, Layer.InputNodeCount);
//         BiasVelocities = Vector.Create(Layer.OutputNodeCount);
//     }

//     public void Update(Vector nodeValues, LayerSnapshots.Simple snapshot)
//     {
//         foreach (int outputNodeIndex in ..Layer.OutputNodeCount)
//         {
//             foreach(int inputNodeIndex in ..Layer.InputNodeCount)
//             {
//                 // partial derivative cost with respect to weight of current connection
//                 var derivativeCostWrtWeight = snapshot.LastRawInput[inputNodeIndex] * nodeValues[outputNodeIndex];
//                 CostGradientWeights[outputNodeIndex, inputNodeIndex] += derivativeCostWrtWeight;
//             }

//             // derivative cost with respect to bias (bias' = 1)
//             var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
//             CostGradientBiases[outputNodeIndex] += derivativeCostWrtBias;
//         }
//     }

//     public void Apply(int dataCounter)
//     {
//         var averagedLearningRate = Optimizer.LearningRate / dataCounter;
//         var weightDecay = 1 - Optimizer.Regularization * averagedLearningRate; //used against overfitting

//         foreach(int outputNodeIndex in ..Layer.OutputNodeCount)
//         {
//             var biasVelocity = BiasVelocities[outputNodeIndex] * Optimizer.Momentum - CostGradientBiases[outputNodeIndex] * averagedLearningRate;
//             BiasVelocities[outputNodeIndex] = biasVelocity;
//             Layer.Biases[outputNodeIndex] += biasVelocity;
//             CostGradientBiases[outputNodeIndex] = 0;

//             foreach(int inputNodeIndex in ..Layer.InputNodeCount)
//             {
//                 var weight = Layer.Weights[outputNodeIndex, inputNodeIndex];
//                 var weightVelocity = WeightVelocities[outputNodeIndex, inputNodeIndex] * Optimizer.Momentum - CostGradientWeights[outputNodeIndex, inputNodeIndex] * averagedLearningRate;
//                 WeightVelocities[outputNodeIndex, inputNodeIndex] = weightVelocity;
//                 Layer.Weights[outputNodeIndex, inputNodeIndex] = weight * weightDecay + weightVelocity;
//             }
//         }
//     }

//     public void GradientCostReset()
//     {
//         CostGradientBiases.ResetZero();
//         CostGradientWeights.ResetZero();
//     }

//     public void FullReset()
//     {
//         GradientCostReset();

//         BiasVelocities.ResetZero();
//         WeightVelocities.ResetZero();
//     }
// }