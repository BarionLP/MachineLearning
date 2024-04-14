namespace BarionGPT;

public sealed class SGD(double learningRate) {
    public double LearningRate { get; } = learningRate;

    public void Update(Vector<double> predicted, int actual) {
        var loss = -Math.Log(predicted[actual]); 


    }

    //uses a complete set of results
    public static double CrossEntropyLoss(Vector<double> predicted, Vector<double> actual) {
        //double epsilon = 1e-10; // prevent log(0)?
        double loss = 0;

        for(int i = 0; i < predicted.Count; i++) {
            loss -= actual[i] * Math.Log(predicted[i]/* + epsilon*/);
        }

        return loss;
    }
}
