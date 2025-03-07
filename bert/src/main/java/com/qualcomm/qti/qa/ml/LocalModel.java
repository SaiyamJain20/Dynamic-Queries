package com.qualcomm.qti.qa.ml;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LocalModel {

    private Interpreter interpreter;
    private Map<String, Integer> vocab;
    private Map<Integer, String> idToToken;
    private FeatureConverter featureConverter;
    private AssetManager assetManager;

    // Constants
    private static final int MAX_SEQ_LEN = 384;
    private static final int MAX_QUERY_LEN = 64;
    private static final int MAX_SPAN_LENGTH = 30;

    public LocalModel(AssetManager assets) {
        this.assetManager = assets;
        // Load TFLite model
        interpreter = new Interpreter(loadModelFile("mobilebert.tflite"));
        // Load vocabulary and create reverse mapping
        vocab = loadVocab(assetManager, "vocab2.txt");
        idToToken = createIdToTokenMap(vocab);
        // Initialize FeatureConverter with the same parameters as in your Java code
        featureConverter = new FeatureConverter(vocab, true, MAX_QUERY_LEN, MAX_SEQ_LEN);
    }

    // Loads the TFLite model from assets.
    private ByteBuffer loadModelFile(String modelPath) {
        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Loads the vocabulary file from assets.
    private Map<String, Integer> loadVocab(AssetManager assets, String vocabFileName) {
        Map<String, Integer> vocab = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(assets.open(vocabFileName)))) {
            String line;
            int index = 0;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    vocab.put(line, index);
                    index++;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return vocab;
    }

    // Creates a reverse mapping from token id to token.
    private Map<Integer, String> createIdToTokenMap(Map<String, Integer> vocab) {
        Map<Integer, String> idToToken = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            idToToken.put(entry.getValue(), entry.getKey());
        }
        return idToToken;
    }

    /**
     * This function takes a question (and uses a hardcoded context passage) to
     * produce an answer from the model.
     */
    public String getRecommendation(String question, String context) {
        // Convert question and context to model features using FeatureConverter.
        Feature feature = featureConverter.convert(question, context);

        // Convert List<Integer> features to fixed-size arrays.
        int[] inputIdsArray = listToArray(feature.getInputIds(), MAX_SEQ_LEN);
        int[] inputMaskArray = listToArray(feature.getInputMask(), MAX_SEQ_LEN);
        int[] segmentIdsArray = listToArray(feature.getSegmentIds(), MAX_SEQ_LEN);

        // Create batch input arrays (batch size = 1).
        float[][] inputIds = new float[1][MAX_SEQ_LEN];
        float[][] inputMask = new float[1][MAX_SEQ_LEN];
        float[][] segmentIds = new float[1][MAX_SEQ_LEN];

        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            inputIds[0][i] = inputIdsArray[i];
            inputMask[0][i] = inputMaskArray[i];
            segmentIds[0][i] = segmentIdsArray[i];
        }

        // Prepare output arrays for start and end logits.
        float[][] startLogits = new float[1][MAX_SEQ_LEN];
        float[][] endLogits = new float[1][MAX_SEQ_LEN];

        // Create an output map to capture both outputs.
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, startLogits);
        outputMap.put(1, endLogits);

        // Run inference.
        interpreter.runForMultipleInputsOutputs(
                new Object[]{inputIds, inputMask, segmentIds}, outputMap);

        // Select the best span from the logits.
        int[] bestSpan = selectBestSpan(startLogits[0], endLogits[0]);
        int bestStart = bestSpan[0];
        int bestEnd = bestSpan[1];

        // Decode the answer from the token indices.
        return decodeAnswer(feature, bestStart, bestEnd);
    }

    // Helper: Convert a List<Integer> to an int[] of a fixed size (padding with zeros if needed).
    private int[] listToArray(List<Integer> list, int size) {
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = (i < list.size()) ? list.get(i) : 0;
        }
        return array;
    }

    // Selects the best answer span based on start and end logits.
    private int[] selectBestSpan(float[] startLogits, float[] endLogits) {
        float bestScore = Float.NEGATIVE_INFINITY;
        int bestStart = 0;
        int bestEnd = 0;
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            for (int j = i; j < Math.min(i + MAX_SPAN_LENGTH, MAX_SEQ_LEN); j++) {
                float score = startLogits[i] + endLogits[j];
                if (score > bestScore) {
                    bestScore = score;
                    bestStart = i;
                    bestEnd = j;
                }
            }
        }
        return new int[]{bestStart, bestEnd};
    }

    /**
     * Decodes the answer span back into the original text using the token-to-original mapping.
     * An offset of 1 is applied (to account for the [CLS] token).
     */
    private String decodeAnswer(Feature feature, int start, int end) {
        final int OFFSET = 1;
        int shiftedStart = start + OFFSET;
        int shiftedEnd = end + OFFSET;
        Map<Integer, Integer> tokenToOrigMap = feature.getTokenToOrigMap();
        List<String> origTokens = feature.getOrigTokens();
        if (!tokenToOrigMap.containsKey(shiftedStart) || !tokenToOrigMap.containsKey(shiftedEnd)) {
            return "";
        }
        int origStart = tokenToOrigMap.get(shiftedStart);
        int origEnd = tokenToOrigMap.get(shiftedEnd);
        StringBuilder sb = new StringBuilder();
        for (int i = origStart; i <= origEnd && i < origTokens.size(); i++) {
            sb.append(origTokens.get(i));
            if (i < origEnd) {
                sb.append(" ");
            }
        }
        return sb.toString();
    }
}

