using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

namespace MachineLearningToolkit.Utility
{
    class PathNormalizer
    {
        public static string NormalizeDirectory(string path)
        {
            if (Directory.Exists(path))
            {
                return Path.GetFullPath(path);
            }
            else
            {
                return Directory.CreateDirectory(path).FullName;
            }
        }

        public static string NormalizeFilePath(string path)
        {
            return Path.GetFullPath(path);
        }
    }
}
