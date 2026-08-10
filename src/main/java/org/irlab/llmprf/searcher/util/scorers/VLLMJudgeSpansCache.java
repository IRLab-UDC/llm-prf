package org.irlab.llmprf.searcher.util.scorers;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.util.*;

public class VLLMJudgeSpansCache {

  private static final ObjectMapper objectMapper = new ObjectMapper();

  private final String cacheFile;
  private final boolean useNarrative;
  private final Map<String, Entry> cache;
  private BufferedWriter cacheWriter;
  private boolean closed = false;

  private static class Entry {
    final boolean isRelevant;
    final List<String> spans;

    Entry(boolean isRelevant, List<String> spans) {
      this.isRelevant = isRelevant;
      this.spans = spans;
    }
  }

  public VLLMJudgeSpansCache(String cacheDirectory, boolean useNarrative) throws IOException {
    this.useNarrative = useNarrative;
    this.cacheFile = cacheDirectory + (useNarrative ? "/vllm_judgespans_cache.tsv" : "/vllm_judgespans_nonarr_cache.tsv");
    this.cache = new HashMap<>();

    new File(cacheDirectory).mkdirs();

    File file = new File(cacheFile);
    if (file.exists()) {
      try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
        String line;
        while ((line = reader.readLine()) != null) {
          String[] parts = line.split("\t", 4);
          if (parts.length >= 4) {
            String key = parts[0] + "_" + parts[1];
            boolean isRelevant = Boolean.parseBoolean(parts[2]);
            try {
              List<String> spans = objectMapper.readValue(parts[3], new TypeReference<List<String>>() {});
              cache.put(key, new Entry(isRelevant, spans));
            } catch (Exception e) {
              cache.put(key, new Entry(isRelevant, Collections.emptyList()));
            }
          }
        }
      }
    }
    System.out.println("Loaded VLLM judge+spans cache with " + cache.size() + " entries from " + cacheFile);

    this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
  }

  private Entry getOrFetch(int queryId, int docId, String queryText, String narrative, String docText) throws IOException {
    String key = queryId + "_" + docId;

    Entry cached = cache.get(key);
    if (cached != null) {
      return cached;
    }

    VLLMJudgeSpansScorer.Result result = VLLMJudgeSpansScorer.judgeAndExtractSpans(
        queryText, useNarrative ? narrative : null, docText);

    Entry entry = new Entry(result.isRelevant, result.spans);

    String spansJson = objectMapper.writeValueAsString(result.spans);
    cacheWriter.write(String.format("%d\t%d\t%s\t%s\n", queryId, docId, result.isRelevant, spansJson));
    cacheWriter.flush();

    cache.put(key, entry);
    return entry;
  }

  public LLMCache asLLMCache() {
    return new LLMCache() {
      @Override
      public LLMResult get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException {
        Entry entry = getOrFetch(queryId, docId, queryText, narrative, docText);
        double placeholder = entry.isRelevant ? 1.0 : 0.0;
        return new LLMResult(entry.isRelevant, placeholder, placeholder);
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
        VLLMJudgeSpansCache.this.close();
      }
    };
  }

  public TermsProvider asTermsProvider() {
    return new TermsProvider() {
      @Override
      public List<String> get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException {
        Entry entry = cache.get(queryId + "_" + docId);
        return entry != null ? entry.spans : Collections.emptyList();
      }

      @Override
      public boolean containsQuery(int queryId) {
        String prefix = queryId + "_";
        return cache.keySet().stream().anyMatch(k -> k.startsWith(prefix));
      }

      @Override
      public void close() throws IOException {
        VLLMJudgeSpansCache.this.close();
      }
    };
  }

  public void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    if (cacheWriter != null) {
      cacheWriter.close();
    }
  }
}
