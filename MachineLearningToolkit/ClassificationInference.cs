using System;
using System.Collections.Generic;

namespace MachineLearningToolkit
{
    public class ClassificationInference
    {
        public string DateTime { get; set; }
        public string ImagePath { get; set; }
        public Dictionary<string, float> Classifications { get; set; }
    }
}