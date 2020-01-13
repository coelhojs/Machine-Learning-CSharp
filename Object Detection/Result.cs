using System.Collections.Generic;

namespace MachineLearningToolkit
{
    public class Result
    {
        public int NumDetections { get; set; }
        public List<DetectionInference> Results { get; set; }

    }
}
