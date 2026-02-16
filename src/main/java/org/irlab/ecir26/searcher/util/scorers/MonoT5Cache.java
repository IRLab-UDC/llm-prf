package org.irlab.ecir26.searcher.util.scorers;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Cache manager for MonoT5 scorer results.
 * Manages persistent cache with all MonoT5 metrics.
 */
public class MonoT5Cache implements LLMCache {
  private final String cacheDir;
  private final String cacheFile;

  private Map<String, LLMResult> cache;
  private BufferedWriter cacheWriter;

  public MonoT5Cache(String cacheDirectory) throws IOException {
    this.cacheDir = cacheDirectory;
    this.cacheFile = cacheDir + "/t5_cache.tsv";

    new File(cacheDir).mkdirs();
    this.cache = new HashMap<>();

    // Load existing cache: query_id \t doc_id \t prediction \t logit_true \t logit_false \t prob_true \t prob_false \t score
    File cacheFileObj = new File(cacheFile);
    if (cacheFileObj.exists()) {
      try (BufferedReader reader = new BufferedReader(new FileReader(cacheFileObj))) {
        String line;
        while ((line = reader.readLine()) != null) {
          String[] parts = line.split("\t");
          if (parts.length >= 8) {
            String cacheKey = parts[0] + "_" + parts[1];
            String prediction = parts[2];
            boolean isRelevant = "true".equalsIgnoreCase(prediction);
            double probTrue = Double.parseDouble(parts[5]);
            double score = Double.parseDouble(parts[7]);
            cache.put(cacheKey, new LLMResult(isRelevant, probTrue, score));
          }
        }
      }
    }
    System.out.println("Loaded MonoT5 cache with " + cache.size() + " entries from " + cacheFile);
    // Open for appending
    this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
  }

  @Override
  public LLMResult get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException {
    String cacheKey = queryId + "_" + docId;

    if (cache.containsKey(cacheKey)) {
      return cache.get(cacheKey);
    }

    // Not in cache, evaluate with MonoT5
    MonoT5Scorer.MonoT5Result monoResult = MonoT5Scorer.evaluate(queryText, docText);

    // Write to cache file: query_id \t doc_id \t prediction \t logit_true \t logit_false \t prob_true \t prob_false \t score
    cacheWriter.write(String.format("%d\t%d\t%s\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n",
                                    queryId,
                                    docId,
                                    monoResult.prediction,
                                    monoResult.logitTrue,
                                    monoResult.logitFalse,
                                    monoResult.probTrue,
                                    monoResult.probFalse,
                                    monoResult.score));
    cacheWriter.flush();

    // Convert to LLMResult and store in memory cache
    LLMResult result = new LLMResult(monoResult.isRelevant, monoResult.probTrue, monoResult.score);
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
