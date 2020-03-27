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

    [Header ("FACS Head Movement Action Units ")]

    [Range (-1f, 1f)] // Point #2 & #31
    public float horizontalHeadOrientation = 0f;
    public float horizontalHeadOrientationScale = 0.05f;
    float neutralHorizontalHeadOrientation = 0f;

    [Range (-1f, 1f)] // Point #28 & #29
    public float verticalHeadOrientation = 0f;
    public float verticalHeadOrientationScale = 0.05f;
    float neutralVerticalHeadOrientation = 0f;

    [Header ("FACS Facial Action Units")]

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

    public float trackingInterval = 0.25f;
    public float trackingSmoothingTime = 0.1f;

    public TextAsset yamlFile;
    public TextAsset cascadeModel;

    [Header ("Debugging")]

    public Image debugImage;
    public bool displayOffsetMarkers = true;
    public bool displayCalibrationMarkers = true;
    public bool displaySmoothedPositions = true;

    // Internal

    WebCamTexture webcamTexture;
    int width = 0;
    int height = 0;

    String filePath;
    String fmFilePath;

    FacemarkLBFParams fParams;
    FacemarkLBF facemark;

    Texture2D convertedTexture;

    VectorOfVectorOfPointF landmarks;
    VectorOfVectorOfPointF lastPositions = null;

    VectorOfRect facesVV;

    Vector3[] calibratedPositions = new Vector3[68];
    Vector3[] smoothedPositions = new Vector3[68];
    Vector3 noseOffset;
    bool calibrated = false;
    bool recording = false;

    Vector3[] velRef = new Vector3[68];

    void Start () {
        //cascadePath = Path.Combine (Directory.GetCurrentDirectory (), AssetDatabase.GetAssetPath (cascadeFile));

        // We initialize webcam texture data
        webcamTexture = new WebCamTexture ();
        webcamTexture.Play ();

        width = webcamTexture.width;
        height = webcamTexture.height;

        // We store settings internally for openCV after loading them in, these are the filepaths
        filePath = Path.Combine (Application.persistentDataPath, cascadeModel.name + ".xml");
        fmFilePath = Path.Combine (Application.persistentDataPath, "lbfmodel.yaml");

        // We initialize the facemark system that will be used to recognize our face
        fParams = new FacemarkLBFParams ();
        fParams.ModelFile = fmFilePath;
        facemark = new FacemarkLBF (fParams);
        facemark.LoadModel (fParams.ModelFile);

        File.WriteAllBytes (filePath, cascadeModel.bytes);

        convertedTexture = new Texture2D (width, height);

        Debug.Log ("Tracking Started! Recording with " + webcamTexture.deviceName + " at " + webcamTexture.width + "x" + webcamTexture.height);

        InvokeRepeating ("Track", trackingInterval, trackingInterval);

    }

    void UpdateActionUnits () {

        if (calibrated) {

            // Horizontal Head Orientation
            horizontalHeadOrientation = Mathf.Clamp ((neutralHorizontalHeadOrientation - Vector3.Distance (calibratedPositions[30], smoothedPositions[1])) * horizontalHeadOrientationScale, 0f, 1f);
            // Vertical Head Orientation
            verticalHeadOrientation = Mathf.Clamp ((neutralVerticalHeadOrientation - Vector3.Distance (calibratedPositions[57], smoothedPositions[8])) * verticalHeadOrientationScale, 0f, 1f);
        }

    }

    void Update () {
        // Calibrate - you need to do this at least once
        if (Input.GetKeyDown (KeyCode.Space)) {
            Calibrate ();
        }

        // Calculate moothed positions

        if (recording) {

            for (int i = 0; i < 68; i++) {
                Vector3 currentVector = new Vector3 (landmarks[0][i].X, landmarks[0][i].Y * -1f, 0f);
                smoothedPositions[i] = Vector3.SmoothDamp (smoothedPositions[i], currentVector, ref velRef[i], trackingSmoothingTime);

                // Draw Smoothed positions
                if (displaySmoothedPositions) {
                    Debug.DrawLine (smoothedPositions[i], smoothedPositions[i] + (Vector3.forward * 3f), UnityEngine.Color.white);
                }
            }

            UpdateActionUnits ();

        }

    }

    void Calibrate () {

        // We get landmark positions
        for (int i = 0; i < 68; i++) {
            Vector3 markerPos = new Vector3 (landmarks[0][i].X, landmarks[0][i].Y * -1f, 0f);
            calibratedPositions[i] = markerPos;
        }

        // Horizontal Head Orientation
        neutralHorizontalHeadOrientation = Vector3.Distance (calibratedPositions[30], smoothedPositions[1]);
        // Vertical Head Orientation
        neutralVerticalHeadOrientation = Vector3.Distance (calibratedPositions[57], smoothedPositions[8]);

        calibrated = true;
        Debug.Log ("Calibrated Sucessfully!");

    }

    void Track () {
        if (lastPositions != null) {
            lastPositions = landmarks;
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
                    noseOffset = new Vector3 (landmarks[0][67].X, landmarks[0][67].Y * -1f, 0f);

                    // We draw markers and computer positions
                    for (int j = 0; j < 68; j++) {

                        Vector3 markerPos = new Vector3 (landmarks[0][j].X, landmarks[0][j].Y * -1f, 0f);

                        if (displayOffsetMarkers) {
                            Debug.DrawLine (markerPos, markerPos + (Vector3.forward * 3f), UnityEngine.Color.green, trackingInterval);
                        }

                        AdjustCalibration (j, markerPos);

                    }
                    recording = true;
                } else {
                    recording = false;
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
    void AdjustCalibration (int i, Vector3 markerPos) {
        if (calibrated) {

            calibratedPositions[i] += noseOffset - calibratedPositions[67];

        }

    }

    // Display calibration in-scene
    void DisplayCalibration () {
        if (calibratedPositions[0] != Vector3.zero) {
            for (int i = 0; i < calibratedPositions.Length; i++) {
                Debug.DrawLine (calibratedPositions[i], calibratedPositions[i] + (Vector3.forward * 3f), UnityEngine.Color.red, trackingInterval);
            }
        }
    }

}