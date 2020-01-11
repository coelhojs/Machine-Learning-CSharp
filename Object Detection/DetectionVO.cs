using System.Drawing;

namespace MachineLearningToolkit.ObjectDetection
{
    public class DetectionVO
    {
        public Rectangle BoundingBox { get; set; }

        public string Class { get; set; }

        public string ImagePath { get; set; }

        public float Score { get; set; }
    }
}