package com.example.modeloclasificadortensorflowia;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.modeloclasificadortensorflowia.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.TensorFlowLite;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    ArrayList<String> permisosNoAprobados;
    TextView txtResults;
    ImageView mImageView;
    Bitmap mSelectedImage;
    Button btnCamara, btnGaleria, btnReconocimiento;

    String[] labels;

    private static final int IMAGE_SIZE = 224;
    public static int REQUEST_CAMERA = 111;
    public static int REQUEST_GALLERY = 222;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        txtResults = findViewById(R.id.txtresults);
        mImageView = findViewById(R.id.image_view);
        btnCamara=findViewById(R.id.btCamera);
        btnGaleria=findViewById(R.id.btGallery);
        btnReconocimiento=findViewById(R.id.btnReconocer);

        labels=new String[1001];
        int cnt=0;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String linea =  bufferedReader.readLine();
            while(linea!=null){
                labels[cnt]=linea;
                cnt++;
                linea =  bufferedReader.readLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }



        ArrayList<String> permisos_requeridos = new ArrayList<String>();
        permisos_requeridos.add(Manifest.permission.CAMERA);
        permisos_requeridos.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        permisos_requeridos.add(Manifest.permission.MANAGE_EXTERNAL_STORAGE);

        permisosNoAprobados = getPermisosNoAprobados(permisos_requeridos);
        requestPermissions(permisosNoAprobados.toArray(new String[permisosNoAprobados.size()]), 100);



    }

    public void abrirGaleria (View view){
        Intent i = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, REQUEST_GALLERY);
    }
    public void abrirCamera (View view){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CAMERA);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && null != data) {
            try {
                if (requestCode == REQUEST_CAMERA)
                    mSelectedImage = (Bitmap) data.getExtras().get("data");

                else
                    mSelectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());

                if(mSelectedImage!=null){
                    mSelectedImage = displayImageAndClassify(mSelectedImage);
                }

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private Bitmap displayImageAndClassify(Bitmap image) {
        int dimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
        mImageView.setImageBitmap(image);

        image = Bitmap.createScaledBitmap(image, IMAGE_SIZE, IMAGE_SIZE, false);
        return image;
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        for(int i=0; i<permissions.length; i++){
            if(permissions[i].equals(Manifest.permission.CAMERA)){
                btnCamara.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            }
        }
    }

    public ArrayList<String> getPermisosNoAprobados(ArrayList<String> listaPermisos) {
        ArrayList<String> list = new ArrayList<String>();
        boolean habilitado;

        if (Build.VERSION.SDK_INT >= 23) {
            for (String permiso : listaPermisos) {
                if (checkSelfPermission(permiso) != PackageManager.PERMISSION_GRANTED) {
                    list.add(permiso);
                    habilitado = false;
                } else {
                    habilitado = true;
                }

                if (permiso.equals(Manifest.permission.CAMERA)) {
                    btnCamara.setEnabled(habilitado);
                }
            }
        }

        return list;
    }

    public void reconocer (View view){
        try {

            ByteBuffer byteBuffer = getByteBufferFromImage(mSelectedImage);
            ModelUnquant model = ModelUnquant.newInstance(MainActivity.this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, IMAGE_SIZE, IMAGE_SIZE, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            txtResults.setText(labels[getMax(outputFeature0.getFloatArray())] + " ");
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private ByteBuffer getByteBufferFromImage(Bitmap image) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
        int pixel = 0;
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return byteBuffer;
    }

    int getMax(float[] arr)
    {
        int max=0;
        for (int i=0;i<arr.length;i++ ){
            if(arr[i]>arr[max]) max=i;

        }
        return max;
    }


}