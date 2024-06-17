package simpleSampleData;

import java.io.*;
import java.util.Random;

public class SyntheticMNISTGenerator {

    private static final int IMAGE_SIZE = 28;
    private static final int NUMBER_OF_IMAGES = 100;
    private static final int NOISE_LEVEL = 220; // Adjust noise level as needed

    public static void main(String[] args) throws IOException {
        int[][] leftToRightImages = generateLeftToRightDiagonalImages(NUMBER_OF_IMAGES);
        int[][] rightToLeftImages = generateRightToLeftDiagonalImages(NUMBER_OF_IMAGES);

        int totalImages = NUMBER_OF_IMAGES * 2;
        int[][] allImages = new int[totalImages][IMAGE_SIZE * IMAGE_SIZE];
        int[] allLabels = new int[totalImages];

        // Combine both types of images into one array
        for (int i = 0; i < NUMBER_OF_IMAGES; i++) {
            allImages[i] = leftToRightImages[i];
            allLabels[i] = 0; // Label for left-to-right diagonal images
            allImages[i + NUMBER_OF_IMAGES] = rightToLeftImages[i];
            allLabels[i + NUMBER_OF_IMAGES] = 1; // Label for right-to-left diagonal images
        }

        // Scramble the order of the images and labels
        scrambleOrder(allImages, allLabels);


        String imageFile = "C:\\temp\\synthetic_images.idx3-ubyte";
               String labelFile = "C:\\temp\\synthetic_labels.idx1-ubyte";


        writeMNISTFile(imageFile, allImages);
        writeMNISTLabelFile(labelFile, allLabels);

        System.out.println("Synthetic MNIST files with noise generated successfully.");
    }

    private static int[][] generateLeftToRightDiagonalImages(int numberOfImages) {
        Random random = new Random();
        int[][] images = new int[numberOfImages][IMAGE_SIZE * IMAGE_SIZE];
        for (int n = 0; n < numberOfImages; n++) {
            for (int row = 0; row < IMAGE_SIZE; row++) {
                for (int col = 0; col < IMAGE_SIZE; col++) {
                    int baseValue = (row == col) ? 255 : 0; // Diagonal line from top-left to bottom-right
                    int noise = random.nextInt(NOISE_LEVEL * 2 + 1) - NOISE_LEVEL;
                    int pixelValue = clamp(baseValue + noise, 0, 255);
                    images[n][row * IMAGE_SIZE + col] = pixelValue;
                }
            }
        }
        return images;
    }

    private static int[][] generateRightToLeftDiagonalImages(int numberOfImages) {
        Random random = new Random();
        int[][] images = new int[numberOfImages][IMAGE_SIZE * IMAGE_SIZE];
        for (int n = 0; n < numberOfImages; n++) {
            for (int row = 0; row < IMAGE_SIZE; row++) {
                for (int col = 0; col < IMAGE_SIZE; col++) {
                    int baseValue = (row + col == IMAGE_SIZE - 1) ? 255 : 0; // Diagonal line from top-right to bottom-left
                    int noise = random.nextInt(NOISE_LEVEL * 2 + 1) - NOISE_LEVEL;
                    int pixelValue = clamp(baseValue + noise, 0, 255);
                    images[n][row * IMAGE_SIZE + col] = pixelValue;
                }
            }
        }
        return images;
    }

    private static void scrambleOrder(int[][] images, int[] labels) {
        Random random = new Random();
        for (int i = images.length - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);
            int[] tempImage = images[index];
            images[index] = images[i];
            images[i] = tempImage;

            int tempLabel = labels[index];
            labels[index] = labels[i];
            labels[i] = tempLabel;
        }
    }

    private static void writeMNISTFile(String fileName, int[][] images) throws IOException {
    	
       DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fileName)));
            int magicNumber = 0x00000803; // Magic number for MNIST image files
            int numberOfImages = images.length;
            int numberOfRows = IMAGE_SIZE;
            int numberOfColumns = IMAGE_SIZE;

            dos.writeInt(magicNumber);
            dos.writeInt(numberOfImages);
            dos.writeInt(numberOfRows);
            dos.writeInt(numberOfColumns);

            for (int[] image : images) {
                for (int pixel : image) {
                    dos.writeByte(pixel);
                }
        }
            
        dos.flush();  dos.close();
    }

    private static void writeMNISTLabelFile(String fileName, int[] labels) throws IOException 
    {
        DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fileName)));
        int magicNumber = 0x00000801; // Magic number for MNIST label files
        int numberOfLabels = labels.length;

        dos.writeInt(magicNumber);
        dos.writeInt(numberOfLabels);

        for (int label : labels) 
        {
        	dos.writeByte(label);
         }
        
        dos.flush();  dos.close();
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
