using System;
using Machine_Learning;

namespace Object_Detection
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var detection = new ObjectDetection();
            detection.Run();
        }
    }
}
