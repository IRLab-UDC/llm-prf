package org.irlab.llmprf.searcher.util.scorers;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class VLLMCache implements LLMCache {
  private final String cacheDir;
  private final String cacheFile;

  private Map<String, LLMResult> cache;
  private BufferedWriter cacheWriter;

  public VLLMCache(String cacheDirectory) throws IOException {
    this(cacheDirectory, "vllm_cache.tsv");
  }

  public VLLMCache(String cacheDirectory, String cacheFileName) throws IOException {
    this.cacheDir = cacheDirectory;
    this.cacheFile = cacheDir + "/" + cacheFileName;

    new File(cacheDir).mkdirs();
    this.cache = new HashMap<>();

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
            cache.put(cacheKey, new LLMResult(isRelevant, probTrue, probTrue));
          }
        }
      }
    }
    System.out.println("Loaded VLLM cache with " + cache.size() + " entries from " + cacheFile);

    this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
  }

  @Override
  public LLMResult get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException {
    String cacheKey = queryId + "_" + docId;

    if (cache.containsKey(cacheKey)) {
      return cache.get(cacheKey);
    }

    VLLMScorer.VLLMResult vllmResult = VLLMScorer.evaluate(queryText, narrative, docText);

    cacheWriter.write(String.format("%d\t%d\t%s\t%.16f\t%.16f\n",
                                    queryId,
                                    docId,
                                    vllmResult.isRelevant,
                                    vllmResult.probTrue,
                                    vllmResult.probFalse));
    cacheWriter.flush();

    LLMResult result = new LLMResult(vllmResult.isRelevant, vllmResult.probTrue, vllmResult.probTrue);
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
