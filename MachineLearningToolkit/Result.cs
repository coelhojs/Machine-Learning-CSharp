using System;
using System.Collections.Generic;

namespace MachineLearningToolkit
{
    public class Result
    {
        public DateTime DateTime { get; set; }
        public int NumDetections { get; set; }
        public List<DetectionInference> Results { get; set; }

    }
}
