using System.Drawing;

namespace MachineLearningToolkit.ObjectDetection
{
    public class Inference
    {
        public Rectangle BoundingBox { get; set; }
        public string Class { get; set; }
        public string Image { get; set; }
        public float Score { get; set; }
    }
}