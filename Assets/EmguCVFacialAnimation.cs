using System;
using System.Collections;
using System.Drawing;
using System.IO;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using UnityEngine;
using UnityEngine.UI;

public class EmguCVFacialAnimation : MonoBehaviour {

    WebCamTexture webcamTexture;
    int width = 0;
    int height = 0;

    String fileName = "haarcascade_frontalface_alt2";
    String filePath;

    String fmFileName = "lbfmodel";
    String fmFilePath;

    FacemarkLBFParams fParams;
    FacemarkLBF facemark;

    TextAsset yamlFile;
    TextAsset cascadeModel;

    public Image outputImage;

    Texture2D convertedTexture;

    void Start () {
        // We initialize webcam texture data
        webcamTexture = new WebCamTexture ();
        webcamTexture.Play ();

        print (webcamTexture.deviceName);

        width = webcamTexture.width;
        height = webcamTexture.height;

        filePath = Path.Combine (Application.persistentDataPath, fileName + ".xml");
        fmFilePath = Path.Combine (Application.persistentDataPath, fmFileName + ".yaml");

        fParams = new FacemarkLBFParams ();
        fParams.ModelFile = fmFilePath;
        // https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/ normally 68
        fParams.NLandmarks = 68; // number of landmark points
        fParams.InitShapeN = 10; // number of multiplier for make data augmentation
        fParams.StagesN = 5; // amount of refinement stages
        fParams.TreeN = 6; // number of tree in the model for each landmark point
        fParams.TreeDepth = 5; //he depth of decision tree
        facemark = new FacemarkLBF (fParams);

        cascadeModel = Resources.Load<TextAsset> (fileName);
        File.WriteAllBytes (filePath, cascadeModel.bytes);
        yamlFile = Resources.Load<TextAsset> (fmFileName);

        convertedTexture = new Texture2D (width, height);

    }

    // Use this for initialization
    void Update () {

        convertedTexture.SetPixels (webcamTexture.GetPixels ());
        convertedTexture.Apply ();

        UMat img = new UMat ();
        TextureConvert.Texture2dToOutputArray (convertedTexture, img);
        CvInvoke.Flip (img, img, FlipType.Vertical);

        using (CascadeClassifier classifier = new CascadeClassifier (filePath))
        using (UMat gray = new UMat ()) {
            CvInvoke.CvtColor (img, gray, ColorConversion.Bgr2Gray);

            Rectangle[] faces = null;
            try {
                faces = classifier.DetectMultiScale (gray);

                // We can drag detection rectangles on the debug output
                //      foreach (Rectangle face in faces) {
                //       CvInvoke.Rectangle (img, face, new MCvScalar (0, 255, 0));
                //     }

                VectorOfRect facesVV = new VectorOfRect (classifier.DetectMultiScale (gray));
                VectorOfVectorOfPointF landmarks = new VectorOfVectorOfPointF ();
                facemark.LoadModel (fParams.ModelFile);

                if (facemark.Fit (gray, facesVV, landmarks)) {
                    Rectangle[] facesRect = faces.ToArray ();
                    for (int i = 0; i < facesRect.Length; i++) {
                        FaceInvoke.DrawFacemarks (img, landmarks[i], new MCvScalar (0, 255, 0));
                    }
                }
            } catch (Exception e) {
                Debug.Log (e.Message);
                return;
            }
        }

        Texture2D texture = TextureConvert.InputArrayToTexture2D (img, FlipType.Vertical);

        // Laggy, dont
        // ResizeTexture (texture);
        RenderTexture (texture);
    }

    private void RenderTexture (Texture2D texture) {

        outputImage.sprite = Sprite.Create (texture, new Rect (0, 0, texture.width, texture.height), new Vector2 (0.5f, 0.5f));
    }

    private void ResizeTexture (Texture2D texture) {

        var transform = outputImage.rectTransform;
        transform.sizeDelta = new Vector2 (texture.width, texture.height);
        transform.position = new Vector3 (-texture.width / 2, -texture.height / 2);
        transform.anchoredPosition = new Vector2 (0, 0);
    }
}