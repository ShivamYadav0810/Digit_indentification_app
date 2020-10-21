package com.example.mnist_android;
import android.graphics.BitmapFactory;
import android.view.View;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;
import android.graphics.Bitmap;
import android.util.Log;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;



public class MainActivity extends AppCompatActivity {

    // UI elements
    ImageView imageView;
    TextView textView;

    // Variables for communicating with the model file
    static {
        System.loadLibrary("tensorflow_inference");
    }
    //Model file giving where my freezed Mnist file is present
    private static final String MODEL_FILE = "file:///android_asset/optimized_frozen_mnist_model.pb";
    //input node having same name as that of input file in mnist code
    private static final String INPUT_NODE = "x_input";
    //this array depicts shape of our input data which says that we are going to analyse one image
    //at a time so one array of 784 elements because it is the exact no. of pixels that we will be
    //dealing with (28*28)pixels
    private static final int[] INPUT_SHAPE = {1, 784};
    //output node having same name as of our variable having actual values of output from mnist code
    private static final String OUTPUT_NODE = "y_actual";
   //it os going to be used to be actually perform the inference by connecting it to the graph
    private TensorFlowInferenceInterface inferenceInterface;

    // Variables to help hold the images in our drawable folder and iterate through the list
    private int imageListIndex = 9;
    private final int[] imageIDList = {
            R.drawable.digit0,
            R.drawable.digit1,
            R.drawable.digit2,
            R.drawable.digit3,
            R.drawable.digit4,
            R.drawable.digit5,
            R.drawable.digit6,
            R.drawable.digit7,
            R.drawable.digit8,
            R.drawable.digit9
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Set up the UI elements
        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView) findViewById(R.id.text_view);

        // Initialize the inference variable to use our mnist model
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    // Function to call when user presses on predict button
    // Calls the code to perform the prediction
    public void predictDigitClick(View view) {
        // Get the image data as a float array
        float[] pixelBuffer = convertImage();
        // Get the label that represents the prediction
        float[] results = formPrediction(pixelBuffer);
//        for (float result : results) {
//            Log.d("result", String.valueOf(result));
//        }
        printResults(results);
    }
//our aim is to find the max and second max value from our output model list
    private void printResults(float[] results) {
        float max = 0;
        float secondMax = 0;
        int maxIndex = 0;
        int secondMaxIndex = 0;
        for(int i = 0; i < 10; i++) {
            if (results[i] > max) {
                secondMax = max;
                secondMaxIndex = maxIndex;
                max = results[i];
                maxIndex = i;
            } else if (results[i] < max && results[i] > secondMax) {
                secondMax = results[i];
                secondMaxIndex = i;
            }
        }
        String output = "Model predicts: " + String.valueOf(maxIndex) +
                ", second choice: " + String.valueOf(secondMaxIndex);
        textView.setText(output);
    }

    // Function to actually make the prediction
    // Takes in array of floats that represents the image data
    // Outputs an array of floats that represents the label based on the current prediction
    private float[] formPrediction(float[] pixelBuffer) {
        // Fill the input node with the pixel buffer
        //filenodeflow is used to fit in float values in node
        //here we take our input node according to the shape and its populating it with
        // the value of pixels
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE, pixelBuffer);
        //creating array of output node
        //this is going to run the inference on the back end its going to take our model that was
        // stored in our inference interface  its going to all the actual prediction its going to
        //output the prediction value into our outpude node which itself is not a Sting array and
        // just an  array of the nodes that we once intp
        // Make the prediction by running inference on our model and store results in output node
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        //generates list of lables as our result
        float[] results = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        // Store value of output node (results) into a float array
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        //passing this for tobe used in formPrediction function
        return results;
    }

    // Function to convert currently displayed image into a float array to feed into the model
    // Returns a float array for input into the model
    //now we are about to ensure that the image we are using is of correct format
//so we use preduct digital click
//we want to take some image data so an image founded our dueable folder
//or our drawalbe array and convert it to a bitmap
//then wwe want to reize and reshape our bitmap and then we want to return that bitmap as a float array
//which we can fit in our model and our model takes in an array of floats specifically one array at a time of 784 elements
//so we need to flatten out our image once it converts to bitmap and then wreturn as a flows array
    private float[] convertImage() {
        // Convert current image to a scaled 28 x 28 bitmap
        //creating bitmap object and then passing the exact image index as a type of specific resource
        Bitmap imageBitmap = BitmapFactory.decodeResource(getResources(),
                imageIDList[imageListIndex]);
        //esnsuring dimentions of image
        imageBitmap = Bitmap.createScaledBitmap(imageBitmap, 28, 28, true);
        //displaying editted image (a/c to our dimentions requirement
        imageView.setImageBitmap(imageBitmap);
        //our mnist code requires code in float array format and this is array of bitmaps
        //so we need to convert this shit into int and then in turn to float array
        int[] imageAsIntArray = new int[784];
        float[] imageAsFloatArray = new float[784];
        // Get the pixel values of the bitmap and store them in a flattened int array
        //basically its gonna take our image bitmap and its gonna get the pixels fromthat and
        // convert that to a interface so it will actually store the results in our iages into
        // right here
        imageBitmap.getPixels(imageAsIntArray, 0, 28, 0, 0, 28, 28);
        // Convert the int array into a float array
        for (int i = 0; i < 784; i++) {
            imageAsFloatArray[i] = imageAsIntArray[i] / -16777216;
        }
        return imageAsFloatArray;
    }

    // Function to call when user presses on load next image button
    // Calls the code to load and display the next image in the image view
    public void loadNextImageClick(View view) {
        // Increase the index counter basically defines how to iterate between the images
        if (imageListIndex >= 9) {
            imageListIndex = 0;
        } else {
            imageListIndex += 1;
        }
        // imageListIndex = (imageListIndex >= 9) ? 0 : imageListIndex + 1;
        // Load the image found at imageListIndex from our imageIDList
        //basically dsiplaying rhe images by calling upon out images
        imageView.setImageDrawable(getDrawable(imageIDList[imageListIndex]));
    }

}
//now we are about to ensure that the image we are using is of correct format
//so we use preduct digital click
//we want to take some image data so an image founded our dueable folder
//or our drawalbe array and convert it to a bitmap
//then wwe want to reize and reshape our bitmap and then we want to return that bitmap as a float array
//which we can fit in our model and our model takes in an array of floats specifically one array at a time of 784 elements
//so we need to flatten out our image once it converts to bitmap and then wreturn as a flows array