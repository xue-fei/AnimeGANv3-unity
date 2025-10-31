using UnityEditor;
using UnityEngine;

public class EditorTool : Editor
{
    [MenuItem("工具/打开沙盒路径")]
    static void OpenPersistentDataPath()
    {
        System.Diagnostics.Process.Start(@Application.persistentDataPath);
    }
}