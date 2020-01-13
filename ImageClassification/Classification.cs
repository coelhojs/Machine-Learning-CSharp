using System.Collections.Generic;

namespace MachineLearningToolkit.ImageClassification
{
    public class Classification
    {
        public string Image { get; set; }
        public Dictionary<string, float> Classifications { get; set; }
    }
}