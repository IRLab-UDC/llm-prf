package org.irlab.ecir26.searcher.util.scorers;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.util.Map;

/**
 * Simple helper to call the MonoT5 Flask service running on localhost:5000
 * Expects the service at POST /eval with JSON {"query":..., "document":...}
 * and a JSON response containing all evaluation metrics.
 */
public class MonoT5Scorer {

  private static final String SERVICE_URL = "http://localhost:5000/eval";
  private static final ObjectMapper objectMapper = new ObjectMapper();

  /**
   * Result from MonoT5 evaluation with all metrics
   */
  public static class MonoT5Result {
    public final boolean isRelevant;
    public final double logitTrue;
    public final double logitFalse;
    public final double probTrue;
    public final double probFalse;
    public final double score;
    public final String prediction;

    public MonoT5Result(boolean isRelevant, double logitTrue, double logitFalse, double probTrue, double probFalse,
                        double score, String prediction) {
      this.isRelevant = isRelevant;
      this.logitTrue = logitTrue;
      this.logitFalse = logitFalse;
      this.probTrue = probTrue;
      this.probFalse = probFalse;
      this.score = score;
      this.prediction = prediction;
    }
  }

  /**
   * Evaluate document relevance and return full result with all metrics
   */
  public static MonoT5Result evaluate(String query, String document) {
    try {
      URL url = URI.create(SERVICE_URL).toURL();
      HttpURLConnection con = (HttpURLConnection) url.openConnection();
      con.setRequestMethod("POST");
      con.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
      con.setDoOutput(true);

      // Create JSON payload using Jackson
      ObjectNode payload = objectMapper.createObjectNode();
      payload.put("query", query);
      payload.put("document", document);

      try (OutputStream os = con.getOutputStream()) {
        byte[] input = objectMapper.writeValueAsBytes(payload);
        os.write(input);
      }

      int code = con.getResponseCode();
      InputStreamReader isr = new InputStreamReader(code >= 200
                                                    && code < 300 ? con.getInputStream() : con.getErrorStream(),
                                                    "utf-8");
      try (BufferedReader br = new BufferedReader(isr)) {
        StringBuilder resp = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
          resp.append(line);
        }
        return parseResponse(resp.toString());
      }
    } catch (Exception e) {
      System.err.println("MonoT5Scorer error: " + e.getMessage());
      return new MonoT5Result(false, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, 0.0, 0.0, 0.0, "false");
    }
  }


  private static MonoT5Result parseResponse(String json) {
    try {
      Map<String, Object> response = objectMapper.readValue(json, new TypeReference<>() {
      });

      String prediction = (String) response.getOrDefault("prediction", "false");
      boolean isRelevant = "true".equalsIgnoreCase(prediction);

      double logitTrue = getDoubleValue(response, "logit_true");
      double logitFalse = getDoubleValue(response, "logit_false");
      double probTrue = getDoubleValue(response, "prob_true");
      double probFalse = getDoubleValue(response, "prob_false");
      double score = getDoubleValue(response, "score");

      return new MonoT5Result(isRelevant, logitTrue, logitFalse, probTrue, probFalse, score, prediction);

    } catch (Exception e) {
      System.err.println("Error parsing MonoT5 response: " + e.getMessage());
      return new MonoT5Result(false, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, 0.0, 0.0, 0.0, "false");
    }
  }

  private static double getDoubleValue(Map<String, Object> map, String key) {
    Object value = map.get(key);
    if (value instanceof Number) {
      return ((Number) value).doubleValue();
    }
    return Double.NEGATIVE_INFINITY;
  }
}
