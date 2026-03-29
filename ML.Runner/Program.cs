using System.Globalization;
using ML.Runner.Samples.Mnist;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

// var random = Random.Shared;
var random = new Random(69);

MnistModel.Run(random);

#if DEBUG
// forces all remaining finalizers to be called to make sure all have been disposed
GC.Collect();
#endif