/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System;
using System.IO;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using static Tensorflow.Binding;

namespace Machine_Learning
{
    public class ObjectDetection
    {
        public float MIN_SCORE = 0.5f;

        string ModelDir = "models\\vera_poles_trees\\1";
        string ImageDir = "images";
        string GraphFile = "frozen_inference_graph.pb";
        string LabelFile = "label_map.pbtxt";
        string PicFile = "input.png";

        public ObjectDetection(){

        }

        NDArray imgArr;

        public bool Run()
        {
            // read in the input image
            imgArr = ReadTensorFromImageFile(Path.Join(ImageDir, PicFile));

            var graph = ImportGraph();

            using (var sess = tf.Session(graph))
                Predict(sess);

            return true;
        }

        public Graph ImportGraph()
        {
            var graph = new Graph().as_default();
            graph.Import(Path.Join(ModelDir, GraphFile));

            return graph;
        }

        public void Predict(Session sess)
        {
            var graph = tf.get_default_graph();

            Tensor tensorNum = graph.OperationByName("num_detections");
            Tensor tensorBoxes = graph.OperationByName("detection_boxes");
            Tensor tensorScores = graph.OperationByName("detection_scores");
            Tensor tensorClasses = graph.OperationByName("detection_classes");
            Tensor imgTensor = graph.OperationByName("image_tensor");
            Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };

            var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));

            buildOutputImage(results);
        }

        private NDArray ReadTensorFromImageFile(string file_name)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.read_file(file_name, "file_reader");
            var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            var casted = tf.cast(decodeJpeg, TF_DataType.TF_UINT8);
            var dims_expander = tf.expand_dims(casted, 0);

            using (var sess = tf.Session(graph))
                return sess.run(dims_expander);
        }

        private void buildOutputImage(NDArray[] resultArr)
        {
            // get pbtxt items
            PbtxtItems pbTxtItems = PbtxtParser.ParsePbtxtFile(Path.Join(ModelDir, LabelFile));

            // get bitmap
            Bitmap bitmap = new Bitmap(Path.Join(ImageDir, PicFile));

            var scores = resultArr[2].AsIterator<float>();
            var boxes = resultArr[1].GetData<float>();
            var id = np.squeeze(resultArr[3]).GetData<float>();
            for (int i=0; i< scores.size; i++)
            {
                float score = scores.MoveNext();
                if (score > MIN_SCORE)
                {
                    float top = boxes[i * 4] * bitmap.Height;
                    float left = boxes[i * 4 + 1] * bitmap.Width;
                    float bottom = boxes[i * 4 + 2] * bitmap.Height;
                    float right = boxes[i * 4 + 3] * bitmap.Width;

                    Rectangle rect = new Rectangle()
                    {
                        X = (int)left,
                        Y = (int)top,
                        Width = (int)(right - left),
                        Height = (int)(bottom - top)
                    };

                    string name = pbTxtItems.items.Where(w => w.id == id[i]).Select(s=>s.display_name).FirstOrDefault();

                    drawObjectOnBitmap(bitmap, rect, score, name);
                }
            }

            string path = Path.Join(ImageDir, "output.jpg");
            bitmap.Save(path);
            Console.WriteLine($"Processed image is saved as {path}");
        }

        private void drawObjectOnBitmap(Bitmap bmp, Rectangle rect, float score, string name)
        {
            using (Graphics graphic = Graphics.FromImage(bmp))
            {
                graphic.SmoothingMode = SmoothingMode.AntiAlias;
                
                using (Pen pen = new Pen(Color.Red, 2))
                {
                    graphic.DrawRectangle(pen, rect);

                    Point p = new Point(rect.Right + 5, rect.Top + 5);
                    string text = string.Format("{0}:{1}%", name, (int)(score * 100));
                    graphic.DrawString(text, new Font("Verdana", 8), Brushes.Red, p);
                }
            }
        }
    }
}
