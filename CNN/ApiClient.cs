// ApiClient.cs

using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.IO;

// APIからのJSONレスポンスを格納するためのクラス
[System.Serializable]
public class ClassificationResponse
{
    public string place;
    public string time;
}

public class ApiClient : MonoBehaviour
{
    // FlaskサーバーのURL
    private string apiUrl = "http://127.0.0.1:5000/classify"; // Pythonを動かしているPCのIPアドレスにすることもあります

    // UIボタンなどから呼び出す
    public void StartClassification()
    {
        // ここでファイル選択ダイアログを開き、画像パスを取得する
        // 例として、ここではPC上の固定パスを使います
        string imagePath = "C:/path/to/your/test_image.jpg";

        if (File.Exists(imagePath))
        {
            StartCoroutine(UploadAndClassify(imagePath));
        }
        else
        {
            Debug.LogError("ファイルが見つかりません: " + imagePath);
        }
    }

    private IEnumerator UploadAndClassify(string imagePath)
    {
        // 画像ファイルをバイト配列として読み込む
        byte[] imageData = File.ReadAllBytes(imagePath);

        // POSTリクエストを作成
        WWWForm form = new WWWForm();
        // 'image'というキーで画像データを追加します。このキー名はFlask側と一致させる必要があります。
        form.AddBinaryData("image", imageData, Path.GetFileName(imagePath), "image/jpeg");

        using (UnityWebRequest www = UnityWebRequest.Post(apiUrl, form))
        {
            Debug.Log("リクエストを送信中...");
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("レスポンス受信成功!");
                string jsonResponse = www.downloadHandler.text;
                Debug.Log("受信したJSON: " + jsonResponse);

                // JSONをパースして結果を取得
                ClassificationResponse response = JsonUtility.FromJson<ClassificationResponse>(jsonResponse);
                
                Debug.Log($"場所: {response.place}, 時間帯: {response.time}");

                // --- ここで、結果に応じたシーン生成処理を呼び出す ---
                // LoadSceneByCategoryAndTime(response.place, response.time);
            }
            else
            {
                Debug.LogError("リクエスト失敗: " + www.error);
            }
        }
    }
}