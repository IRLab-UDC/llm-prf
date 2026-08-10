package org.irlab.llmprf.searcher.util.scorers;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class VLLMJudgeSpansScorer {

  private static final String SERVICE_URL = "http://" + System.getenv().getOrDefault("VLLM_HOST", "localhost:8080") + "/judge_spans";
  private static final ObjectMapper objectMapper = new ObjectMapper();

  public static class Result {
    public final boolean isRelevant;
    public final List<String> spans;

    public Result(boolean isRelevant, List<String> spans) {
      this.isRelevant = isRelevant;
      this.spans = spans;
    }
  }

  public static Result judgeAndExtractSpans(String query, String narrative, String document) {
    try {
      URL url = URI.create(SERVICE_URL).toURL();
      HttpURLConnection con = (HttpURLConnection) url.openConnection();
      con.setRequestMethod("POST");
      con.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
      con.setDoOutput(true);

      String processedDocument = document.replaceAll("\\s+", " ").trim();
      processedDocument = processedDocument.replaceAll("[|\\-]+", " ");

      String prompt = String.format(
          "You are an expert TREC assessor. Your task is to judge relevance and, if the "
          + "document is relevant, identify the passages that most directly address the query.\n\n"
          + "Instructions:\n"
          + "\t1. Read the query carefully.\n"
          + "\t2. Read the document.\n"
          + "\t3. Decide if the document provides information that answers or helps address the query.\n"
          + "\t4. If it is relevant, identify the passages or sentences that directly address "
          + "the query. Return only text that appears in the document.\n"
          + "\t5. If it is not relevant, return an empty list of spans.\n\n"
          + "Query: %s\n\n"
          + (narrative != null && !narrative.isEmpty() ? String.format("Assessor instructions:\n%s\n\n", narrative) : "")
          + "Document:\n%s\n",
          query.trim(), processedDocument);

      ObjectNode payload = objectMapper.createObjectNode();
      payload.put("prompt", prompt);

      try (OutputStream os = con.getOutputStream()) {
        os.write(objectMapper.writeValueAsBytes(payload));
      }

      int code = con.getResponseCode();
      InputStreamReader isr = new InputStreamReader(
          code >= 200 && code < 300 ? con.getInputStream() : con.getErrorStream(), "utf-8");
      try (BufferedReader br = new BufferedReader(isr)) {
        StringBuilder resp = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) resp.append(line);
        return parseResponse(resp.toString());
      }
    } catch (Exception e) {
      System.err.println("VLLMJudgeSpansScorer error: " + e.getMessage());
      return new Result(false, Collections.emptyList());
    }
  }

  @SuppressWarnings("unchecked")
  private static Result parseResponse(String json) {
    try {
      Map<String, Object> response = objectMapper.readValue(json, new TypeReference<>() {});
      boolean isRelevant = Boolean.TRUE.equals(response.get("relevant"));
      Object spansObj = response.get("spans");
      List<String> spans = spansObj instanceof List ? (List<String>) spansObj : Collections.emptyList();
      return new Result(isRelevant, spans);
    } catch (Exception e) {
      System.err.println("VLLMJudgeSpansScorer parse error: " + e.getMessage());
      return new Result(false, Collections.emptyList());
    }
  }
}
