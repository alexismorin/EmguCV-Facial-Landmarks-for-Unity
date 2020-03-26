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

    public float trackingInverval = 0.25f;

    public TextAsset yamlFileUser;
    public TextAsset cascadeModelUser;

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

    FacemarkLBFParams fParams;
    FacemarkLBF facemark;

    TextAsset yamlFile;
    TextAsset cascadeModel;

    Texture2D convertedTexture;

    VectorOfVectorOfPointF landmarks;
    VectorOfRect facesVV;

    Vector3[] calibratedPositions = new Vector3[68];
    Vector3 nosePosition;

    bool calibrated = false;

    void Start () {
        // We initialize webcam texture data
        webcamTexture = new WebCamTexture ();
        webcamTexture.Play ();

        width = webcamTexture.width;
        height = webcamTexture.height;

        filePath = Path.Combine (Application.persistentDataPath, fileName + ".xml");
        fmFilePath = Path.Combine (Application.persistentDataPath, fmFileName + ".yaml");

        fParams = new FacemarkLBFParams ();
        fParams.ModelFile = fmFilePath;
        fParams.NLandmarks = 68; // number of landmark points, https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
        fParams.InitShapeN = 10; // number of multiplier for make data augmentation
        fParams.StagesN = 5; // amount of refinement stages
        fParams.TreeN = 6; // number of tree in the model for each landmark point
        fParams.TreeDepth = 5; //he depth of decision tree
        facemark = new FacemarkLBF (fParams);
        facemark.LoadModel (fParams.ModelFile);

        cascadeModel = Resources.Load<TextAsset> (fileName);
        File.WriteAllBytes (filePath, cascadeModel.bytes);
        yamlFile = Resources.Load<TextAsset> (fmFileName);

        convertedTexture = new Texture2D (width, height);

        Debug.Log ("Tracking Started! Recording with " + webcamTexture.deviceName + " at " + webcamTexture.width + "x" + webcamTexture.height);

        //     InvokeRepeating ("Track", trackingInverval, trackingInverval);

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

        calibrated = true;
        Debug.Log ("Calibrated Sucessfully!");

    }

    void Update () {

        // Calibrate - you need to do this at least once
        if (Input.GetKeyDown (KeyCode.Space)) {
            Calibrate ();
        }

        // We fetch webcam texture data
        convertedTexture.SetPixels (webcamTexture.GetPixels ());
        convertedTexture.Apply ();

        // We convert the webcam texture2D into the OpenCV image format
        UMat img = new UMat ();
        TextureConvert.Texture2dToOutputArray (convertedTexture, img);
        CvInvoke.Flip (img, img, FlipType.Vertical);

        using (CascadeClassifier classifier = new CascadeClassifier (filePath)) {
            using (UMat gray = new UMat ()) {

                // We convert the OpenCV image format to the facial detection API parsable monochrome image type and detect the faces
                CvInvoke.CvtColor (img, gray, ColorConversion.Bgr2Gray);
                facesVV = new VectorOfRect (classifier.DetectMultiScale (gray));
                landmarks = new VectorOfVectorOfPointF ();

                // we fit facial landmarks onto the face data
                if (facemark.Fit (gray, facesVV, landmarks)) {

                    FaceInvoke.DrawFacemarks (img, landmarks[0], new MCvScalar (255, 255, 0, 255));

                    // We calculate the nose position to use as a capture center
                    nosePosition = new Vector3 (landmarks[0][30].X, landmarks[0][30].Y * -1f, 0f);

                    // We draw markers and computer positions
                    for (int j = 0; j < 68; j++) {

                        Vector3 markerPos = new Vector3 (landmarks[0][j].X, landmarks[0][j].Y * -1f, 0f);
                        AdjustCalibration (markerPos, j);
                        UpdateActionUnits ();

                        if (displayOffsetMarkers) {
                            Debug.DrawLine (markerPos, markerPos + (Vector3.forward * 3f), UnityEngine.Color.green);
                        }

                    }
                }

                if (displayCalibrationMarkers) {
                    DisplayCalibration ();
                }
            }
        }

        // We render out the calculation result into the debug image
        if (debugImage) {
            Texture2D texture = TextureConvert.InputArrayToTexture2D (img, FlipType.Vertical);
            debugImage.sprite = Sprite.Create (texture, new Rect (0, 0, texture.width, texture.height), new Vector2 (0.5f, 0.5f));
        }
    }

    // Adjusts the calibration  based on the nose offset
    void AdjustCalibration (Vector3 markerPos, int index) {
        if (calibrated) {
            calibratedPositions[index] += nosePosition - calibratedPositions[30];
        }
    }

    // Display calibration in-scene
    void DisplayCalibration () {
        if (calibratedPositions[0] != Vector3.zero) {
            for (int i = 0; i < calibratedPositions.Length; i++) {
                Debug.DrawLine (calibratedPositions[i], calibratedPositions[i] + (Vector3.forward * 3f), UnityEngine.Color.red);
            }
        }
    }

}