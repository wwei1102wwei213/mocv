package com.wei.mocv;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import org.bytedeco.javacpp.Loader;


import org.bytedeco.javacpp.opencv_core;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.samples.facedetect.DetectionBasedTracker;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.WeakReference;

import static org.bytedeco.javacpp.helper.opencv_imgproc.cvCalcHist;
import static org.bytedeco.javacpp.opencv_core.CV_HIST_ARRAY;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_COMP_CORREL;
import static org.bytedeco.javacpp.opencv_imgproc.cvCompareHist;
import static org.bytedeco.javacpp.opencv_imgproc.cvNormalizeHist;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    static {
        System.loadLibrary("detection_based_tracker");
    }

    private static Exception loadingException = null;
    public static void tryLoad() throws Exception {
        if (loadingException != null) {
            throw loadingException;
        } else {
            try {
                Loader.load(org.bytedeco.javacpp.opencv_core.class);
                Loader.load(org.bytedeco.javacpp.opencv_core.class);
                Loader.load(org.bytedeco.javacpp.opencv_core.class);
                Loader.load(org.bytedeco.javacpp.opencv_core.class);
            } catch (Throwable t) {

            }
        }
    }

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Load native library after(!) OpenCV initialization
                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    private MyHandler mHandler;
    private ImageView iv1, iv2;
    private TextView tv, tv_clear;
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);
        mHandler = new MyHandler(this);
        iv1 = findViewById(R.id.iv1);
        iv2 = findViewById(R.id.iv2);
        tv = findViewById(R.id.tv);
        tv_clear = findViewById(R.id.tv_clear);
        tv_clear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                SaveFlag = false;
            }
        });
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    private boolean SaveFlag = false;
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        }
        if (facesArray.length>0) {
            if (!SaveFlag) {
                SaveFlag = saveImage(mRgba, facesArray[0], "TestCV");
            } else {
                saveImage(mRgba, facesArray[0], "TestCV_temp");
            }

        }

        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }

    private static final String PATH_CMP =  Environment.getExternalStorageDirectory() + "/FaceDetect/TestCV.jpg";
    private static final String PATH_TEMP =  Environment.getExternalStorageDirectory() + "/FaceDetect/TestCV_temp.jpg";
    private double cmp_score = 0;
    /**
     * 特征保存
     *
     * @param image    Mat
     * @param rect     人脸信息
     * @param fileName 文件名字
     * @return 保存是否成功
     */
    public boolean saveImage(Mat image, Rect rect, final String fileName) {
        try {
            final String PATH = Environment.getExternalStorageDirectory() + "/FaceDetect/" + fileName + ".jpg";
            // 把检测到的人脸重新定义大小后保存成文件
            Mat sub = image.submat(rect);
            Mat mat = new Mat();
            Size size = new Size(100, 100);
            Imgproc.resize(sub, mat, size);
            File file = new File(PATH);
            if (file.exists()) {
                file.delete();
            } else {
                if (!file.getParentFile().exists()) {
                    file.getParentFile().mkdirs();
                }
            }
            Log.e(TAG, "createNewFile======>"+file.createNewFile()+"\nPATH======>"+PATH);
            Imgcodecs.imwrite(PATH, mat);
            if (!fileName.equals("TestCV")) {
                cmp_score = CmpPic(PATH_CMP, PATH_TEMP);
            }
            mHandler.post(new Runnable() {
                @Override
                public void run() {
                    try {
                        if (fileName.equals("TestCV")) {
                            iv1.setImageBitmap(getLoacalBitmap(PATH));
                            tv.setText("0");
                        } else {
                            iv2.setImageBitmap(getLoacalBitmap(PATH));
                            tv.setText(""+cmp_score);
                        }
                    } catch (Exception e){
                        Log.e(TAG, e.getMessage());
                    }
                }
            });
//            HighGui.imshow(PATH, mat);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 加载本地图片
     * @param url
     * @return
     */
    public static Bitmap getLoacalBitmap(String url) {
        try {
            FileInputStream fis = new FileInputStream(url);
            return BitmapFactory.decodeStream(fis);  ///把流转化为Bitmap图片

        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 特征对比
     *
     * @param
     * @param
     * @return 相似度
     */
    public double CmpPic(String file1, String file2) {

        int l_bins = 20;
        int hist_size[] = {l_bins};

        float v_ranges[] = {0, 100};
        float ranges[][] = {v_ranges};

        opencv_core.IplImage Image1 = cvLoadImage(file1, CV_LOAD_IMAGE_GRAYSCALE);
        opencv_core.IplImage Image2 = cvLoadImage(file2, CV_LOAD_IMAGE_GRAYSCALE);

        opencv_core.IplImage imageArr1[] = {Image1};
        opencv_core.IplImage imageArr2[] = {Image2};

        opencv_core.CvHistogram Histogram1 = opencv_core.CvHistogram.create(1, hist_size, CV_HIST_ARRAY, ranges, 1);
        opencv_core.CvHistogram Histogram2 = opencv_core.CvHistogram.create(1, hist_size, CV_HIST_ARRAY, ranges, 1);

        cvCalcHist(imageArr1, Histogram1, 0, null);
        cvCalcHist(imageArr2, Histogram2, 0, null);

        cvNormalizeHist(Histogram1, 100.0);
        cvNormalizeHist(Histogram2, 100.0);

        return cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL);
    }

    private static class MyHandler extends Handler  {

        private WeakReference<FdActivity> weak;

        public MyHandler(FdActivity activity) {
            weak = new WeakReference<>(activity);
        }

        @Override
        public void handleMessage(Message msg) {
            if (weak.get()!=null) {

            }
        }
    }
}
