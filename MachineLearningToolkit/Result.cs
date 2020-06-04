using System;
using System.Collections.Generic;

namespace MachineLearningToolkit
{
    public class Result
    {
        public string DateTime { get; set; }
        public KeyValuePair<string, string> Error { get; set; }
        public int NumDetections { get; set; }
        public List<DetectionInference> Results { get; set; }

    }
}
