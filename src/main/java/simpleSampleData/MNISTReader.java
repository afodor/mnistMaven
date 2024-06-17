package simpleSampleData;

import java.io.*;
import java.awt.*;
import javax.swing.*;
import java.awt.image.BufferedImage;

/*
 * ChatGPT generated code
 */
public class MNISTReader {

	private static final String IMAGE_FILE = "C:\\temp\\synthetic_images.idx3-ubyte";
	private static final String LABEL_FILE  = "C:\\temp\\synthetic_labels.idx1-ubyte";

	// private static final String IMAGE_FILE = "C:\\Users\\afodor\\git\\LearningExamples\\src\\resources\\train-images-idx3-ubyte";
	 //  private static final String LABEL_FILE = "C:\\Users\\afodor\\git\\LearningExamples\\src\\resources\\train-labels-idx1-ubyte";

    private static int[][] images;
    private static int[] labels;

    public static void main(String[] args) throws IOException {
        images = readImages(IMAGE_FILE);
        labels = readLabels(LABEL_FILE);

        // Display the first 10 images in a single frame
        displayImages(images, labels, 10);
    }

    private static int[][] readImages(String file) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));

        int magicNumber = dis.readInt();
        int numberOfImages = dis.readInt();
        System.out.println("Got " + numberOfImages);
        int rows = dis.readInt();
        int cols = dis.readInt();

        int[][] images = new int[numberOfImages][rows * cols];

        for (int i = 0; i < numberOfImages; i++) {
            for (int j = 0; j < rows * cols; j++) {
                images[i][j] = dis.readUnsignedByte();
            }
        }

        dis.close();
        return images;
    }

    private static int[] readLabels(String file) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));

        int magicNumber = dis.readInt();
        int numberOfLabels = dis.readInt();

        int[] labels = new int[numberOfLabels];

        for (int i = 0; i < numberOfLabels; i++) {
            labels[i] = dis.readUnsignedByte();
        }

        dis.close();
        return labels;
    }

    private static void displayImages(int[][] images, int[] labels, int numImages) {
        JFrame frame = new JFrame("MNIST Digits");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(1000, 1000);
        frame.setLocation(200, 200);
        frame.setLayout(new GridLayout(2, 5));

        for (int i = 0; i < numImages; i++) {
            BufferedImage bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int row = 0; row < 28; row++) {
                for (int col = 0; col < 28; col++) {
                    int gray = 255 - images[i][row * 28 + col];
                    int color = (gray << 16) | (gray << 8) | gray;
                    bufferedImage.setRGB(col, row, color);
                }
            }
            JLabel label = new JLabel(new ImageIcon(bufferedImage));
            label.setText(Integer.toString(labels[i]));
            label.setHorizontalTextPosition(JLabel.CENTER);
            label.setVerticalTextPosition(JLabel.BOTTOM);
            frame.add(label);
        }

        frame.pack();
        frame.setVisible(true);
    }
}
