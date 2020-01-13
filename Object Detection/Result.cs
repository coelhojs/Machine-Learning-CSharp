using System.Collections.Generic;

namespace MachineLearningToolkit.ObjectDetection
{
    public class Result
    {
        public int NumDetections { get; set; }
        public List<Inference> Results { get; set; }

    }
}
