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

    // https://imotions.com/blog/facial-action-coding-system/ A great Action Unit reference

    // Front-Facing

    [Header ("FACS Facial Action Units")]

    [Range (-1f, 1f)] // Point #2 & #31
    public float horizontalHeadOrientation = 0f;
    float neutralHorizontalHeadOrientationPointSpacing;

    [Range (-1f, 1f)] // Point #28 & #29
    public float verticalHeadOrientation = 0f;
    float neutralVerticalHeadOrientationPointSpacing;

    [Header ("FACS Head Movement Action Units ")]

    [Range (-1f, 1f)] // Point #22 & #23
    public float innerBrowRaiser = 0f;

    [Range (-1f, 1f)] // Point #18 & #27
    public float outerBrowRaiser = 0f;

    [Range (-1f, 1f)] // Point #32 & #36
    public float noseWrinkler = 0f;

    [Range (-1f, 1f)] // Point #9
    public float chinRaiser = 0f;

    [Range (-1f, 1f)] // Point #27
    public float mouthStretch = 0f;

    [Range (-1f, 1f)] // Point #59 & #57
    public float lowerLipDepressor = 0f;

    [Range (-1f, 1f)] // Point #14
    public float dimpler = 0f;

    [Range (-1f, 1f)] // Point #39 & 44
    public float blink = 0f;

    [Header ("OpenCV Settings")]
    //   public TextAsset cascadeFile;

    [Header ("Debugging")]

    public Image debugImage;
    public bool displayOffsetMarkers = true;
    public bool displayCalibrationMarkers = true;

    // Internal

    WebCamTexture webcamTexture;
    int width = 0;
    int height = 0;

    String fileName = "haarcascade_frontalface_alt2";
    String filePath;

    String fmFileName = "lbfmodel";
    String fmFilePath;

    String cascadePath;

    FacemarkLBFParams fParams;
    FacemarkLBF facemark;

    TextAsset yamlFile;
    TextAsset cascadeModel;

    Texture2D convertedTexture;

    VectorOfVectorOfPointF landmarks;
    VectorOfRect facesVV;

    Vector3[] calibratedPositions = new Vector3[68];

    void Start () {

        // We initialize webcam texture data
        webcamTexture = new WebCamTexture ();
        webcamTexture.Play ();

        width = webcamTexture.width;
        height = webcamTexture.height;

        //   cascadePath = Path.Combine (Directory.GetCurrentDirectory (), AssetDatabase.GetAssetPath (cascadeFile));

        filePath = Path.Combine (Application.persistentDataPath, fileName + ".xml");
        fmFilePath = Path.Combine (Application.persistentDataPath, fmFileName + ".yaml");

        fParams = new FacemarkLBFParams ();
        fParams.ModelFile = fmFilePath;
        fParams.NLandmarks = 68; // number of landmark points 
        fParams.InitShapeN = 10; // number of multiplier for make data augmentation
        fParams.StagesN = 5; // amount of refinement stages
        fParams.TreeN = 6; // number of tree in the model for each landmark point
        fParams.TreeDepth = 5; //he depth of decision tree
        facemark = new FacemarkLBF (fParams);

        facemark.LoadModel (filePath);

        cascadeModel = Resources.Load<TextAsset> (fileName);
        File.WriteAllBytes (filePath, cascadeModel.bytes);
        yamlFile = Resources.Load<TextAsset> (fmFileName);

        convertedTexture = new Texture2D (width, height);

        Debug.Log ("Tracking Started! Recording with " + webcamTexture.deviceName + " at " + webcamTexture.width + "x" + webcamTexture.height);

        InvokeRepeating ("Evaluate", 0.1f, 0.1f);
    }

    void UpdateActionUnits () {

        // Horizontal Head Orientation
        //   horizontalHeadOrientation =
    }

    void Calibrate () {

        // We get landmark positions
        for (int i = 0; i < 68; i++) {
            Vector3 markerPos = new Vector3 (landmarks[0][i].X, landmarks[0][i].Y * -1f, 0f);
            calibratedPositions[i] = markerPos;
        }

        // We offset them in-reference to the tip of the nose
        for (int i = 0; i < calibratedPositions.Length; i++) {
            calibratedPositions[i] = calibratedPositions[i] - calibratedPositions[30];
            print ("Point #" + i + ": " + calibratedPositions[i]);
        }

        Debug.Log ("Calibrated Sucessfully");

    }

    void Evaluate () {

        if (Input.GetKeyDown (KeyCode.Space)) {
            Calibrate ();
        }

        if (displayCalibrationMarkers) {
            DisplayCalibration ();
        }

        convertedTexture.SetPixels (webcamTexture.GetPixels ());
        convertedTexture.Apply ();

        using (CascadeClassifier classifier = new CascadeClassifier (filePath)) {
            using (UMat gray = new UMat ()) {

                UMat img = new UMat ();

                TextureConvert.Texture2dToOutputArray (convertedTexture, img);
                CvInvoke.Flip (img, img, FlipType.Vertical);

                CvInvoke.CvtColor (img, gray, ColorConversion.Bgr2Gray);

                Rectangle[] faces = classifier.DetectMultiScale (gray);

                facesVV = new VectorOfRect (classifier.DetectMultiScale (gray));
                landmarks = new VectorOfVectorOfPointF ();

                if (facemark.Fit (gray, facesVV, landmarks)) {
                    for (int i = 0; i < faces.Length; i++) {
                        FaceInvoke.DrawFacemarks (img, landmarks[i], new MCvScalar (0, 255, 0));
                        for (int j = 0; j < 68; j++) {
                            if (displayOffsetMarkers) {
                                Vector3 markerPos = new Vector3 (landmarks[i][j].X, landmarks[i][j].Y * -1f, 0f);
                                Debug.DrawLine (markerPos, markerPos + (Vector3.forward * 3f), UnityEngine.Color.yellow);
                            }
                            UpdateActionUnits ();
                        }
                    }
                }

                convertedTexture = TextureConvert.InputArrayToTexture2D (img, FlipType.Vertical);
                debugImage.sprite = Sprite.Create (convertedTexture, new Rect (0, 0, convertedTexture.width, convertedTexture.height), new Vector2 (0.5f, 0.5f));
            }
        }

    }

    void DisplayCalibration () {
        if (calibratedPositions[0] != Vector3.zero) {
            for (int i = 0; i < calibratedPositions.Length; i++) {
                Debug.DrawLine (calibratedPositions[i], calibratedPositions[i] + (Vector3.forward * 3f), UnityEngine.Color.green);
            }
        }
    }

}