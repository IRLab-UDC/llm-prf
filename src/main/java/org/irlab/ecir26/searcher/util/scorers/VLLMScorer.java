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
 * Helper to call the VLLM service running on localhost:8080
 * Expects the service at POST /prob with JSON {"prompt":...}
 * and a JSON response containing {"p_true":..., "p_false":...}
 */
public class VLLMScorer {

  private static final String SERVICE_URL = "http://localhost:8080/prob";
  private static final ObjectMapper objectMapper = new ObjectMapper();

  /**
   * Result from VLLM evaluation with probability scores
   */
  public static class VLLMResult {
    public final boolean isRelevant;
    public final double probTrue;
    public final double probFalse;

    public VLLMResult(boolean isRelevant, double probTrue, double probFalse) {
      this.isRelevant = isRelevant;
      this.probTrue = probTrue;
      this.probFalse = probFalse;
    }
  }

  /**
   * Evaluate document relevance and return full result with probabilities
   *
   * @param query    The search query
   * @param document The document text to evaluate
   * @return VLLMResult containing relevance decision and probabilities
   */
  public static VLLMResult evaluate(String query, String narrative, String document) {
    try {
      URL url = URI.create(SERVICE_URL).toURL();
      HttpURLConnection con = (HttpURLConnection) url.openConnection();
      con.setRequestMethod("POST");
      con.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
      con.setDoOutput(true);
      // We eliminate any line breaks and excessive spaces
      String processedDocument = document.replaceAll("\\s+", " ").trim();
      // we also replace "|" and "-" with " "
      processedDocument = processedDocument.replaceAll("[|\\-]+", " ");
      // Create prompt in the format: [document] ... [query] ... Relevant:
      String prompt = String.format("You are an expert TREC assessor. Your task is to judge relevance.\n\n"
                                    + "Instructions:\n"
                                    + "\t1. Read the query carefully.\n"
                                    + "\t2. Read the document.\n"
                                    + "\t3. Decide if the document provides information that answers or helps address the query.\n"
                                    + "\t4. Respond with 'true' if the document is relevant, or 'false' if it is not.\n\n"
                                    + "Query: %s\n\n"
                                    + (narrative != null && !narrative.isEmpty() ? String.format(
          "Assessor instructions:\n%s\n\n",
          narrative) : "")
                                    + "Document:\n%s\n", query.trim(), processedDocument);

      // Create JSON payload using Jackson
      ObjectNode payload = objectMapper.createObjectNode();
      payload.put("prompt", prompt);

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
      System.err.println("VLLMScorer error: " + e.getMessage());
      return new VLLMResult(false, 0.0, 1.0);
    }
  }

  /**
   * Convenience method: returns only boolean relevance
   * Document is considered relevant if p_true > p_false
   */
  public static boolean isRelevant(String query, String narrative, String document) {
    return evaluate(query, narrative, document).isRelevant;
  }

  /**
   * Parse the JSON response from VLLM service
   * Expected format: {"p_true": 0.679..., "p_false": 0.320...}
   */
  private static VLLMResult parseResponse(String json) {
    try {
      Map<String, Object> response = objectMapper.readValue(json, new TypeReference<>() {
      });

      double probTrue = getDoubleValue(response, "p_true");
      double probFalse = getDoubleValue(response, "p_false");

      // Document is relevant if probability of "true" is higher than "false"
      boolean isRelevant = probTrue > probFalse;

      return new VLLMResult(isRelevant, probTrue, probFalse);

    } catch (Exception e) {
      System.err.println("Error parsing VLLM response: " + e.getMessage());
      return new VLLMResult(false, 0.0, 1.0);
    }
  }

  /**
   * Extract double value from response map, handling various numeric types
   */
  private static double getDoubleValue(Map<String, Object> map, String key) {
    Object value = map.get(key);
    if (value instanceof Number) {
      return ((Number) value).doubleValue();
    }
    return 0.0;
  }
}
