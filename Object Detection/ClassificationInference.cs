using System;
using System.Collections.Generic;

namespace MachineLearningToolkit
{
    public class ClassificationInference
    {
        public DateTime DateTime { get; set; }
        public string Image { get; set; }
        public Dictionary<string, float> Classifications { get; set; }
    }
}