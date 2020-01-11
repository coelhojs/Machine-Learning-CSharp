using System.Collections.Generic;

namespace MachineLearningToolkit.ObjectDetection
{
    public class InferenceResult
    {
        public int NumDetections { get; set; }
        public List<DetectionVO> Results { get; set; }
    }
}
