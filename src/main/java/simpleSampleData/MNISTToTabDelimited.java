package simpleSampleData;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MNISTToTabDelimited {

    public static void main(String[] args) {
        String imagesFilePath = "C:\\temp\\synthetic_images.idx3-ubyte";
        String labelsFilePath = "C:\\temp\\synthetic_labels.idx1-ubyte";
        String outputFilePath = "c:\\temp\\mnist_data.tsv";

        try {
            int[][] images = readImages(imagesFilePath);
            int[] labels = readLabels(labelsFilePath);

            writeTabDelimitedFile(images, labels, outputFilePath);
            System.out.println("Data successfully written to " + outputFilePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int[][] readImages(String filePath) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(filePath));

        // Read magic number and number of images
        dis.readInt();
        int numberOfImages = dis.readInt();
        int numberOfRows = dis.readInt();
        int numberOfColumns = dis.readInt();

        int[][] images = new int[numberOfImages][numberOfRows * numberOfColumns];

        for (int i = 0; i < numberOfImages; i++) {
            for (int j = 0; j < numberOfRows * numberOfColumns; j++) {
                images[i][j] = dis.readUnsignedByte();
            }
        }

        dis.close();
        return images;
    }

    private static int[] readLabels(String filePath) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(filePath));

        // Read magic number and number of labels
        dis.readInt();
        int numberOfLabels = dis.readInt();

        int[] labels = new int[numberOfLabels];

        for (int i = 0; i < numberOfLabels; i++) {
            labels[i] = dis.readUnsignedByte();
        }

        dis.close();
        return labels;
    }

    private static void writeTabDelimitedFile(int[][] images, int[] labels, String filePath) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filePath));

        for (int i = 0; i < images.length; i++) {
            writer.write(labels[i] + "\t");
            for (int j = 0; j < images[i].length; j++) {
                writer.write(images[i][j] + (j == images[i].length - 1 ? "" : "\t"));
            }
            writer.newLine();
        }

        writer.close();
    }
}
