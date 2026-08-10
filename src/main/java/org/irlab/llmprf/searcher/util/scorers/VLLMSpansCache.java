package org.irlab.llmprf.searcher.util.scorers;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.util.*;

public class VLLMSpansCache implements TermsProvider {

  private static final ObjectMapper objectMapper = new ObjectMapper();

  private final String cacheFile;
  private final Map<String, List<String>> cache;
  private final BufferedWriter cacheWriter;
  private final boolean useNarrative;

  public VLLMSpansCache(String cacheDirectory) throws IOException {
    this(cacheDirectory, true);
  }

  public VLLMSpansCache(String cacheDirectory, boolean useNarrative) throws IOException {
    this.useNarrative = useNarrative;
    this.cacheFile = cacheDirectory + (useNarrative ? "/vllm_spans_cache.tsv" : "/vllm_spans_nonarr_cache.tsv");
    this.cache = new HashMap<>();

    new File(cacheDirectory).mkdirs();

    File file = new File(cacheFile);
    if (file.exists()) {
      try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
        String line;
        while ((line = reader.readLine()) != null) {
          String[] parts = line.split("\t", 3);
          if (parts.length >= 3) {
            String key = parts[0] + "_" + parts[1];
            try {
              List<String> spans = objectMapper.readValue(parts[2], new TypeReference<List<String>>() {});
              cache.put(key, spans);
            } catch (Exception e) {
              cache.put(key, Collections.emptyList());
            }
          }
        }
      }
    }
    System.out.println("Loaded VLLM spans cache with " + cache.size() + " entries from " + cacheFile);

    this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
  }

  @Override
  public List<String> get(int queryId, int docId, String queryText, String narrative,
                          String docText) throws IOException {
    String key = queryId + "_" + docId;

    if (cache.containsKey(key)) {
      return cache.get(key);
    }

    List<String> spans = VLLMSpansScorer.getRelevantSpans(queryText, useNarrative ? narrative : null, docText);

    String spansJson = objectMapper.writeValueAsString(spans);
    cacheWriter.write(String.format("%d\t%d\t%s\n", queryId, docId, spansJson));
    cacheWriter.flush();

    cache.put(key, spans);
    return spans;
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
