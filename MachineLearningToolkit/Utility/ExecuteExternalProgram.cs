using NLog;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearningToolkit.Utility
{
    public class ExecuteExternalProgram
    {
        private static Process Process { get; set; }

        private static readonly Logger Log = LogManager.GetCurrentClassLogger();

        public static void SetEnvironmentVariable(string environmentVariablePath)
        {
            var path = Environment.GetEnvironmentVariable("path");
            if (!string.IsNullOrEmpty(environmentVariablePath))
            {
                if (path == null) path = "";


                if (!path.Contains(environmentVariablePath))
                {
                    if (!path.EndsWith(";")) path += ";";

                    path += $@"{environmentVariablePath}";
                    Environment.SetEnvironmentVariable("path", path);
                }
            }
        }

        public static string ExecuteAndGetOutput(string programPath, string arguments, string environmentVariablePath)
        {
            try
            {
                SetEnvironmentVariable(environmentVariablePath);

                var startInfo = new ProcessStartInfo()
                {
                    Arguments = arguments,
                    CreateNoWindow = false,
                    //CreateNoWindow = true,
                    FileName = programPath,
                    RedirectStandardError = true,
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                };
                Process = Process.Start(startInfo);
                Log.Info(Process?.ToString());
                Log.Info("==============================================================");
                Log.Info("Argumentos para execução do app externo: " + arguments);

                do
                {
                    if (!Process.HasExited)
                    {
                        Log.Info(Process.Responding ? "Status = Running" : "Status = Not Responding");
                    }
                } while (!Process.WaitForExit(1000));

                string result = Process.StandardOutput.ReadToEnd();
                if (string.IsNullOrEmpty(result))
                {
                    result = Process?.StandardError.ReadToEnd();
                    Log.Info($"StandardOutput: {result}");
                }
                else
                {
                    Log.Warn($"StandardError: {result}");
                }

                if (!string.IsNullOrEmpty(result))
                {
                    if (result.ToUpper().Contains("ERROR"))
                    {
                        throw new Win32Exception(result);
                    }
                    if (result.ToUpper().Contains("WARNING"))
                    {
                        Log.Warn(result);
                    }
                    else
                    {
                        Log.Info(result);
                    }
                }
                Log.Info("End of process");
                Log.Info("==============================================================");
                return result;
            }
            catch (Exception ex)
            {
                Log.Error(ex);
                throw ex;
            }
            finally
            {
                Process?.Close();
                KillProcess();
            }
        }

        public static int KillProcess()
        {
            try
            {
                if (Process != null && !Process.HasExited)
                {
                    Process.Kill();
                    Log.Info("Process killed");
                }

                Process?.Dispose();
                Process = null;
                return 1;
            }
            catch (Exception ex)
            {
                Log.Info(ex);
                return -1;
            }
        }



    }
}