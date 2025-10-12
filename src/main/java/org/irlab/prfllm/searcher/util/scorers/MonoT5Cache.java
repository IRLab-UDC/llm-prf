package org.irlab.prfllm.searcher.util.scorers;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Cache manager for MonoT5 scorer results.
 * Manages persistent cache with all MonoT5 metrics.
 */
public class MonoT5Cache {
    private final String cacheDir;
    private final String cacheFile;
    
    private Map<String, MonoT5Scorer.MonoT5Result> cache;
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
                        double logitTrue = Double.parseDouble(parts[3]);
                        double logitFalse = Double.parseDouble(parts[4]);
                        double probTrue = Double.parseDouble(parts[5]);
                        double probFalse = Double.parseDouble(parts[6]);
                        double score = Double.parseDouble(parts[7]);
                        cache.put(cacheKey, new MonoT5Scorer.MonoT5Result(
                            isRelevant, logitTrue, logitFalse, probTrue, probFalse, score, prediction));
                    }
                }
            }
        }
        System.out.println("Loaded MonoT5 cache with " + cache.size() + " entries from " + cacheFile);
        // Open for appending
        this.cacheWriter = new BufferedWriter(new FileWriter(cacheFile, true));
    }

    public MonoT5Scorer.MonoT5Result get(int queryId, int docId, String queryText, String docText) throws IOException {
        String cacheKey = queryId + "_" + docId;

        if (cache.containsKey(cacheKey)) {
            return cache.get(cacheKey);
        }

        // Not in cache, evaluate with MonoT5
        MonoT5Scorer.MonoT5Result result = MonoT5Scorer.evaluate(queryText, docText);

        // Write to cache file: query_id \t doc_id \t prediction \t logit_true \t logit_false \t prob_true \t prob_false \t score
        cacheWriter.write(String.format("%d\t%d\t%s\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n",
                queryId, docId, result.prediction, result.logitTrue, result.logitFalse, 
                result.probTrue, result.probFalse, result.score));
        cacheWriter.flush();

        // Store in memory cache
        cache.put(cacheKey, result);

        return result;
    }
    
    /**
     * Returns the number of entries in the cache.
     */
    public int size() {
        return cache.size();
    }
    
    /**
     * Returns true if the cache is empty (no entries loaded).
     */
    public boolean isEmpty() {
        return cache.isEmpty();
    }

    public void close() throws IOException {
        if (cacheWriter != null) {
            cacheWriter.close();
        }
    }
}
