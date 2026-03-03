using ML.Core.Modules;
using ML.Core.Modules.Activations;

var model = new SequenceModule<Vector>
{
    Inner = [
        new PerceptronModule(784, 256) { Activation = new LeakyReLUModule(256) },
        new PerceptronModule(256, 128) { Activation = new LeakyReLUModule(128) },
        new PerceptronModule(128, 10) { Activation = new SoftMaxModule(10) },
    ],
};

var snapshot = model.CreateSnapshot();
var gradients = model.CreateGradients();

var output = model.Forward(Vector.Create(784), snapshot);
var inputGradient = model.Backward(output, snapshot, gradients);

Console.WriteLine(inputGradient);