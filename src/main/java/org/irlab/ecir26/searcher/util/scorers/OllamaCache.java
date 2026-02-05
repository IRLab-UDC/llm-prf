package org.irlab.prfllm.searcher.util.scorers;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Cache manager for Ollama scorer results.
 * Manages persistent cache with model-specific caching support.
 */
public class OllamaCache {
  private final String cacheDir;

  private Map<String, OllamaScorer.OllamaResult> cache;
  private String cacheFile;
  private BufferedWriter cacheWriter;

  public OllamaCache(String cacheDirectory, String modelName) throws IOException {
    this.cacheDir = cacheDirectory;
    new File(cacheDir).mkdirs();
    // Use model name in cache file to support multiple models
    String safeName = modelName.replaceAll("[^a-zA-Z0-9-]", "_");
    this.cacheFile = cacheDir + "/OLLAMA_" + safeName + "_cache.tsv";
    this.cache = new HashMap<>();

    // Load existing cache: query_id \t doc_id \t prediction \t response
    File cacheFileObj = new File(cacheFile);
    if (cacheFileObj.exists()) {
      try (BufferedReader reader = new BufferedReader(new FileReader(cacheFileObj))) {
        String line;
        while ((line = reader.readLine()) != null) {
          String[] parts = line.split("\t", 4); // Limit to 4 parts in case response has tabs
          if (parts.length >= 4) {
            String cacheKey = parts[0] + "_" + parts[1];
            boolean prediction = Boolean.parseBoolean(parts[2]);
            String response = parts[3];
            cache.put(cacheKey, new OllamaScorer.OllamaResult(prediction, response));
          }
        }
      }
    }

    // Open for appending
    this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
  }

  public OllamaScorer.OllamaResult get(int queryId, int docId, String queryText, String docText) throws IOException {
    String cacheKey = queryId + "_" + docId;

    if (cache.containsKey(cacheKey)) {
      return cache.get(cacheKey);
    }

    // Not in cache, evaluate with Ollama
    OllamaScorer.OllamaResult result = OllamaScorer.evaluate(queryText, docText);

    // Write to cache file: query_id \t doc_id \t prediction \t response
    // Clean up response: remove extra whitespace, tabs, newlines
    String cleanedResponse = result.response.replaceAll("\\s+",
                                                        " ")  // Replace all whitespace sequences with single space
                                            .trim();
    cacheWriter.write(String.format("%d\t%d\t%s\t%s\n", queryId, docId, result.isRelevant, cleanedResponse));
    cacheWriter.flush();

    // Store in memory cache
    cache.put(cacheKey, result);

    return result;
  }

  public void close() throws IOException {
    if (cacheWriter != null) {
      cacheWriter.close();
    }
  }
}
