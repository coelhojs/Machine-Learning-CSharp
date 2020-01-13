using System.Collections.Generic;

namespace MachineLearningToolkit
{
    public class ClassificationInference
    {
        public string Image { get; set; }
        public Dictionary<string, float> Classifications { get; set; }
    }
}