package org.irlab.ecir26.searcher.util.scorers;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Cache manager for VLLM scorer results.
 * Manages persistent cache with probability scores.
 */
public class VLLMCache implements LLMCache {
  private final String cacheDir;
  private final String cacheFile;

  private Map<String, LLMResult> cache;
  private BufferedWriter cacheWriter;

  public VLLMCache(String cacheDirectory, String queryType) throws IOException {
    this.cacheDir = cacheDirectory;
    this.cacheFile = cacheDir + "/vllm_cache_" + queryType + ".tsv";

    new File(cacheDir).mkdirs();
    this.cache = new HashMap<>();

    // Load existing cache: query_id \t doc_id \t is_relevant \t prob_true \t prob_false
    File cacheFileObj = new File(cacheFile);
    if (cacheFileObj.exists()) {
      try (BufferedReader reader = new BufferedReader(new FileReader(cacheFileObj))) {
        String line;
        while ((line = reader.readLine()) != null) {
          String[] parts = line.split("\t");
          if (parts.length >= 5) {
            String cacheKey = parts[0] + "_" + parts[1];
            boolean isRelevant = Boolean.parseBoolean(parts[2]);
            double probTrue = Double.parseDouble(parts[3]);
            // For LLMResult, score is probTrue
            cache.put(cacheKey, new LLMResult(isRelevant, probTrue, probTrue));
          }
        }
      }
    }
    System.out.println("Loaded VLLM cache with " + cache.size() + " entries from " + cacheFile);

    // Open for appending
    this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
  }

  @Override
  public LLMResult get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException {
    String cacheKey = queryId + "_" + docId;

    if (cache.containsKey(cacheKey)) {
      return cache.get(cacheKey);
    }

    // Not in cache, evaluate with VLLM
    VLLMScorer.VLLMResult vllmResult = VLLMScorer.evaluate(queryText, narrative, docText);

    // Write to cache file: query_id \t doc_id \t is_relevant \t prob_true \t prob_false
    cacheWriter.write(String.format("%d\t%d\t%s\t%.16f\t%.16f\n",
                                    queryId,
                                    docId,
                                    vllmResult.isRelevant,
                                    vllmResult.probTrue,
                                    vllmResult.probFalse));
    cacheWriter.flush();

    // Convert to LLMResult and store in memory cache
    LLMResult result = new LLMResult(vllmResult.isRelevant, vllmResult.probTrue, vllmResult.probTrue);
    cache.put(cacheKey, result);

    return result;
  }

  /**
   * Returns true if the cache is empty (no entries loaded).
   */
  @Override
  public boolean isEmpty() {
    return cache.isEmpty();
  }

  @Override
  public void close() throws IOException {
    if (cacheWriter != null) {
      cacheWriter.close();
    }
  }
}
