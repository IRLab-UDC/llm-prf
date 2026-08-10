package org.irlab.llmprf.searcher.util.scorers;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

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
    this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
  }

  @Override
  public LLMResult get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException {
    String cacheKey = queryId + "_" + docId;

    if (cache.containsKey(cacheKey)) {
      return cache.get(cacheKey);
    }

    MonoT5Scorer.MonoT5Result monoResult = MonoT5Scorer.evaluate(queryText, docText);

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

    LLMResult result = new LLMResult(monoResult.isRelevant, monoResult.probTrue, monoResult.score);
    cache.put(cacheKey, result);

    return result;
  }

  @Override
  public boolean isEmpty() {
    return cache.isEmpty();
  }

  @Override
  public boolean containsQuery(int queryId) {
    String prefix = queryId + "_";
    return cache.keySet().stream().anyMatch(k -> k.startsWith(prefix));
  }

  @Override
  public void close() throws IOException {
    if (cacheWriter != null) {
      cacheWriter.close();
    }
  }
}
