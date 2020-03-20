using System;
using System.Drawing;

namespace MachineLearningToolkit
{
    public class DetectionInference
    {
        public Rectangle BoundingBox { get; set; }
        public string Class { get; set; }
        public DateTime DateTime { get; set; }
        public string ImagePath { get; set; }
        public float Score { get; set; }
    }
}