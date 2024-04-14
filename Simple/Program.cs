using Simple;

var network = new Network(2, 4, 4, 2);

var trainingContext = new NetworkLearningContext(network);



//Console.WriteLine(data.Replace("\r", "").Replace("\n", "").Length);
//Console.WriteLine(trainingData.Length);

var viewSize = 32;

var trainingData = ConstructTrainingData(1024*8).ToArray();
Console.WriteLine(trainingContext.Cost(trainingData)/trainingData.Length);
//Console.WriteLine();
//WriteModelView(viewSize);

//trainingContext.LearnBatched(trainingData, .1, 128, 128);

var learnRate = .25;
var learnRateDecay = .7;

foreach(var _ in ..1024) {
    //trainingData = ConstructTrainingData(128).ToArray();
    trainingContext.LearnBatched(trainingData, learnRate, 128, 64);
    learnRateDecay *= learnRateDecay;
    //ReconstructView();

    if(_ % 128 == 0) {

        Console.WriteLine();
        Console.WriteLine(trainingContext.Cost(trainingData) / trainingData.Length);
        Console.WriteLine($"Model Succeeded in {trainingContext.Test(trainingData.GetRandomElements(128))*100:F1}% cases");
        Console.WriteLine();
    }

    //Console.WriteLine();
}

//Console.WriteLine(network.Process([6, 1]).Select(d=>d.ToString("F4")).Dump(' '));

Console.WriteLine();
Console.WriteLine("Trained Model:");
WriteModelView(viewSize);
Console.WriteLine();
Console.WriteLine("Actual:");
WriteActualView(viewSize);
Console.WriteLine();


void WriteModelView(int size) {
    foreach (var lineIndex in ..(size/2)) {
        foreach(var charIndex in ..size) {
            var result = network.Process([(Number) charIndex / size, (Number) lineIndex / (size / 2)]);
            //Console.Write($"{result[0]*100:F0} ");
            Console.Write(result[0] > result[1] ? '0' : ' ');
        }
        Console.WriteLine();
    }
}

void WriteActualView(int size) {
    foreach(var lineIndex in ..(size/2)) {
        foreach(var charIndex in ..size) {
            Console.Write(IsInsideShapes((Number) charIndex / size, (Number) lineIndex / (size/2)) ? '0' : ' ');
        }
        Console.WriteLine();
    }
}

static bool IsInsideShapes(double x, double y) {
    // Normalize inputs to the range [-1, 1] for both x and y
    //x = 2 * (x - 0.5);
    //y = 2 * (y - 0.5);

    y = -y;

    // Check if inside circle centered at (0, 0) with radius 0.5
    bool insideCircle = Math.Pow(x, 2) + Math.Pow(y, 2) <= Math.Pow(0.5, 2);

    // Check if inside rectangle from (-0.5, -0.5) to (0.5, 0.5)
    //bool insideRectangle = (x >= -1.0 && x <= 0.0) && (y >= -0.0 && y <= 1.0);

    return insideCircle /*|| insideRectangle*/;
}


IEnumerable<DataPoint<Number[]>> ConstructTrainingData(int count) {
    foreach(var _ in ..count) {
        var x = Random.Shared.NextDouble();
        var y = Random.Shared.NextDouble();
        yield return new DataPoint([x, y],  IsInsideShapes(x, y) ? [1, 0] : [0, 1]);
    }
}

