package com.example.deepfakedetection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Toolbar;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;


import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    ActivityResultLauncher<Intent> imageLauncher;
    ActivityResultLauncher<Intent> cameraLauncher;
    ImageView imageView;
    Button gallery;
    Button capture;
    TextView textView;
    TextView timetextView;
    int cameraRequestCode = 001;
    FaceDetector faceDetector;
    Classifier classifier;
    Spinner modelSpinner;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.textView);
        timetextView = findViewById(R.id.timetextView);
        imageView = findViewById(R.id.imageView);
        gallery = findViewById(R.id.button);
        capture = findViewById(R.id.button2);
        modelSpinner = findViewById(R.id.spinner);


        // 初始化模型選擇 Spinner
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
                R.array.models_array, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(adapter);
        modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String selectedModel = (String) parent.getItemAtPosition(position);
                switch (selectedModel) {
                    case "MobileNetV3":
                        classifier = new Classifier(Utils.assetFilePath(MainActivity.this, "mobilenet_v3_jit.pt"));
                        break;
                    case "ShuffleNetV2":
                        classifier = new Classifier(Utils.assetFilePath(MainActivity.this, "shufflenetV2_jit.pt"));
                        break;
                    case "RepVGG":
                        classifier = new Classifier(Utils.assetFilePath(MainActivity.this, "repvgg_jit.pt"));
                        break;
                    case "RepViT":
                        classifier = new Classifier(Utils.assetFilePath(MainActivity.this, "repvit_jit.pt"));
                        break;
                    case "efficientformerV2":
                        classifier = new Classifier(Utils.assetFilePath(MainActivity.this, "efficientformerV2_jit.pt"));
                        break;
                    case "efficientnet":
                        classifier = new Classifier(Utils.assetFilePath(MainActivity.this, "efficientnet_jit.pt"));
                        break;
                    default:
                        classifier = new Classifier(Utils.assetFilePath(MainActivity.this, "mobilenet_v3_jit.pt"));
                        break;
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // Do nothing
            }
        });
//        classifier = new Classifier(Utils.assetFilePath(this,"mobilenet_v3_jit.pt"));

        FaceDetectorOptions highAccuracyOpts =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                        .build();
        faceDetector = FaceDetection.getClient(highAccuracyOpts);


        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
//                        try {
//                            Uri selectedImage = result.getData().getData();
//                            imageView.setImageURI(selectedImage);
//                            // 從結果中獲取圖像數據
//                            Bitmap imageBitmap = (Bitmap) result.getData().getExtras().get("data");
//                            // 縮放圖像到 224x224
//                            Bitmap scaledBitmap = Bitmap.createScaledBitmap(imageBitmap, 224, 224, true);
//                            // 將圖片顯示在 ImageView 中
//                            imageView.setImageBitmap(scaledBitmap);
//                            String pred = classifier.predict(scaledBitmap);
//                            textView.setText(pred);
//
//                        } catch (Exception e) {
//                            e.printStackTrace();
//                            Toast.makeText(this, "Failed to process the image", Toast.LENGTH_SHORT).show();
//                            }
                        try {
                            Bitmap imageBitmap = (Bitmap) result.getData().getExtras().get("data");

                            processImage(imageBitmap);
                        } catch (Exception e) {
                            e.printStackTrace();
                            Toast.makeText(this, "Failed to process the image", Toast.LENGTH_SHORT).show();
                        }
                    }
                });

        capture.setOnClickListener(v -> {
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            cameraLauncher.launch(cameraIntent);

        });
        // 註冊圖片選擇結果回調
        imageLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri selectedImage = result.getData().getData();
                        imageView.setImageURI(selectedImage);
                        Bitmap image = null;
                        try {
                            image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
//                        image = Bitmap.createScaledBitmap(image, 224, 224, false);

                        processImage(image);
                    }
                }
        );

        // 啟動圖片選擇活動
        gallery.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            imageLauncher.launch(intent);
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
    private void processImage(Bitmap bitmap) {
        imageView.setImageBitmap(bitmap);
        InputImage image = InputImage.fromBitmap(bitmap, 0);
        faceDetector.process(image)
                .addOnSuccessListener(
                        faces -> {
                            if (faces.size() > 0) {
                                Face face = faces.get(0);
                                Bitmap faceBitmap = cropFace(bitmap, face);
                                imageView.setImageBitmap(drawBoundingBox(bitmap, face));
                                predictImage(faceBitmap);
                            } else {
                                textView.setText("No face detected");
                            }
                        })
                .addOnFailureListener(
                        e -> {
                            e.printStackTrace();
                            Toast.makeText(this, "Failed to detect face", Toast.LENGTH_SHORT).show();
                        });
    }

    private Bitmap drawBoundingBox(Bitmap bitmap, Face face) {
//        Bitmap resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
//        Canvas canvas = new Canvas(resultBitmap);
//        Paint paint = new Paint();
//        paint.setColor(Color.RED);
//        paint.setStyle(Paint.Style.STROKE);
//        paint.setStrokeWidth(8);
//
//        android.graphics.Rect bounds = face.getBoundingBox();
//        canvas.drawRect(bounds, paint);
        // 获取人脸的边界框
        android.graphics.Rect bounds = face.getBoundingBox();

        // 增加边界框的大小
        int padding = (bounds.right - bounds.left) / 5; // 调整这个值以增加或减少边界框的大小
        int left = Math.max(0, bounds.left - padding);
        int top = Math.max(0, bounds.top - padding);
        int right = Math.min(bitmap.getWidth(), bounds.right + padding);
        int bottom = Math.min(bitmap.getHeight(), bounds.bottom + padding);
        // 复制原始Bitmap
        Bitmap resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(resultBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(8);

        // 绘制增加后的边界框
        canvas.drawRect(left, top, right, bottom, paint);
        return resultBitmap;
    }

    private Bitmap cropFace(Bitmap bitmap, Face face) {
        android.graphics.Rect bounds = face.getBoundingBox();

        int padding = 20; // 调整这个值以增加或减少边界框的大小
        int left = Math.max(0, bounds.left - padding);
        int top = Math.max(0, bounds.top - padding);
        int right = Math.min(bitmap.getWidth(), bounds.right + padding);
        int bottom = Math.min(bitmap.getHeight(), bounds.bottom + padding);
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top);

//        return Bitmap.createBitmap(bitmap, bounds.left, bounds.top, bounds.width(), bounds.height());
    }

    private void predictImage(Bitmap faceBitmap) {
//        int originalWidth = faceBitmap.getWidth();
//        int originalHeight = faceBitmap.getHeight();
        Bitmap faceBitmapCopy = faceBitmap.copy(faceBitmap.getConfig(), true);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(faceBitmapCopy, 224, 224, true);
        int Height = scaledBitmap.getHeight();
        System.out.println(Height);
        long startTime = System.currentTimeMillis();
        String pred = classifier.predict(scaledBitmap);
        long endTime = System.currentTimeMillis();
        if (TextUtils.isEmpty(pred)) {
            textView.setText("Prediction is empty or null");
        } else {
            textView.setText(pred);
        }
        long predictionTime = endTime - startTime;

//        Bitmap restoredBitmap = Bitmap.createScaledBitmap(scaledBitmap, originalWidth, originalHeight, true);
        timetextView.setText("Prediction Time: " + predictionTime + " ms");
    }

    public static Bitmap deepCopyBitmap(Bitmap originalBitmap) {
        if (originalBitmap == null) {
            return null;
        }

        // 创建一个新的Bitmap
        Bitmap copiedBitmap = Bitmap.createBitmap(
                originalBitmap.getWidth(),
                originalBitmap.getHeight(),
                originalBitmap.getConfig()
        );

        // 创建一个Canvas并将原始Bitmap绘制到新Bitmap上
        Canvas canvas = new Canvas(copiedBitmap);
        canvas.drawBitmap(originalBitmap, 0, 0, null);

        return copiedBitmap;
    }
}