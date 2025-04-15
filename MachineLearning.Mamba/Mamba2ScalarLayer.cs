using MachineLearning.Model.Attributes;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Mamba;

[GeneratedLayer]
public sealed partial class Mamba2ScalarLayer : ILayer<Vector, Mamba2ScalarLayer.Snapshot>
{
    public int MaxSequenceLength /*T*/ => Alpha.Count;
    public int StateDimensions /*N*/ => B.ColumnCount;
    [Weights] public Vector Alpha { get; } // how much memory to keep from the previous step
    [Weights] public Matrix B { get; } // how does the input_t affect the memory h_t
    [Weights] public Matrix C { get; } // how does the memory h_t affect the output_t

    public Mamba2ScalarLayer(int sequenceLength, int stateDimensions)
    {
        Alpha = Vector.Create(sequenceLength);
        B = Matrix.Create(sequenceLength, stateDimensions);
        C = Matrix.Create(sequenceLength, stateDimensions);
    }

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count <= MaxSequenceLength);

        snapshot.Input = input;
        snapshot.Memory.ResetZero();

        for (int t = 0; t < snapshot.SequenceLength; t++)
        {
            // h = alpha_t * h + B_t * x_t
            var h = snapshot.Memory.RowRef(t);
            if (t > 0)
            {
                snapshot.Memory.RowRef(t - 1).MultiplyTo(Alpha[t], h);
            }

            h.AddToSelf(B.RowRef(t).Multiply(input[t])); // add B_t * x_t

            // output[t] = C_t^T * h
            snapshot.Output[t] = C.RowRef(t).Dot(h);
        }

        return snapshot.Output;
    }

    public Vector BackwardPass(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        Debug.Assert(outputGradient.Count <= MaxSequenceLength);

        snapshot.GradientInput.ResetZero();
        snapshot.GradientMemory.ResetZero();

        for (int t = snapshot.SequenceLength - 1; t >= 0; t--)
        {
            // dY = derivative of L wrt outputY[t]
            float dY = outputGradient[t];

            // (a) output[t] = C[t] * H[t]
            // => dC[t] += H[t] * dY
            // => dH[t] += C[t] * dY
            gradients.C.RowRef(t).AddToSelf(snapshot.Memory.RowRef(t).Multiply(dY));
            snapshot.GradientMemory.RowRef(t).AddToSelf(C.RowRef(t).Multiply(dY));

            // (b) h[t] = alpha[t] * h[t-1] + B[t] * inputX[t]
            // => partial wrt alpha[t] = (h[t-1] dot dH[t])
            // => partial wrt h[t-1]  += alpha[t] * dH[t]
            // => partial wrt B[t]    += dH[t] * inputX[t]
            // => partial wrt inputX[t] = (B[t] dot dH[t])

            // derivative w.r.t alpha[t]
            // h[t-1] = (t>0) ? st.h[t-1] : zero
            var hPrev = (t == 0)
                           ? Vector.Create(StateDimensions)
                           : snapshot.Memory.RowRef(t - 1);

            gradients.Alpha[t] += hPrev.Dot(snapshot.GradientMemory.RowRef(t));  // dAlpha

            // derivative w.r.t H[t-1]
            // if t>0, add alpha[t]*dH[t] to dH[t-1]
            if (t > 0)
            {
                snapshot.GradientMemory.RowRef(t - 1).AddToSelf(snapshot.GradientMemory.RowRef(t).Multiply(Alpha[t]));
            }

            // derivative w.r.t. B[t] and input[t]
            // dB[t] = input[t] * dH[t]
            gradients.B.RowRef(t).AddToSelf(snapshot.GradientMemory.RowRef(t).Multiply(snapshot.Input[t]));

            snapshot.GradientInput[t] += B.RowRef(t).Dot(snapshot.GradientMemory.RowRef(t));  // partial w.r.t. input[t]

        }

        return snapshot.GradientInput.Slice(0, snapshot.SequenceLength);
    }

    partial class Snapshot
    {
        public int SequenceLength => Input.Count;
        public Vector Input { get; set; }
        public Vector GradientInput { get; } = Vector.Create(layer.MaxSequenceLength);
        public Vector Output { get; } = Vector.Create(layer.MaxSequenceLength);

        public Matrix Memory /*H*/ { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);
        public Matrix GradientMemory { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<Mamba2ScalarLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(Mamba2ScalarLayer layer)
        {
            float scale = 1.0f / MathF.Sqrt(layer.StateDimensions);

            // affects how much memory the model can keep from the previous step
            // optimally [0.9,1.0] must be [0,1] to prevent vanishing/exploding gradients
            layer.Alpha.Fill(0.9f);

            layer.B.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0f, scale));
            layer.C.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0f, scale));
        }
    }

}
