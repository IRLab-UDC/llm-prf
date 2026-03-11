package org.irlab.llmprf.searcher;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;
import org.irlab.llmprf.searcher.rf.DMM;
import org.irlab.llmprf.searcher.rf.MEDMM;
import org.irlab.llmprf.searcher.rf.RM3;
import org.irlab.llmprf.searcher.rf.RelevanceFeedback;
import org.irlab.llmprf.searcher.smoothing.AdditiveSmoothing;
import org.irlab.llmprf.searcher.smoothing.Smoothing;
import org.irlab.llmprf.searcher.util.StatsProvider;
import org.irlab.llmprf.searcher.util.TRECUtils;
import org.irlab.llmprf.searcher.util.TRECUtils.Topic;
import org.irlab.llmprf.searcher.util.TermWeights;
import org.irlab.llmprf.searcher.util.scorers.LLMCache;
import org.irlab.llmprf.searcher.util.scorers.LLMResult;
import org.irlab.llmprf.searcher.util.scorers.MonoT5Cache;
import org.irlab.llmprf.searcher.util.scorers.VLLMCache;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Stream;

public class TRECSearcherLucene {

  private static final String SEARCH_FIELD = "content";
  private static final String DOCID_FIELD = "docid";
  private static final double smoothingParameter = 0.1d;
  private static final double DMM_LAMBDA = 0.03;
  private static final double MEDMM_LAMBDA = 0.1d;
  private static final double MEDMM_BETA = 1.2d;

  private static final Analyzer analyzer = new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
  private static Map<Integer, Set<Integer>> oracle;

  // ThreadLocal StatsProvider pool - each thread gets its own instance
  private static ThreadLocal<StatsProvider> statsProviderThreadLocal = new ThreadLocal<>();
  private static IndexReader sharedReader; // Shared reader for creating per-thread StatsProviders

  /**
   * Get thread-local StatsProvider. Creates a new one if this thread doesn't have
   * one yet.
   * This allows parallel processing without StatsProvider contention.
   */
  private static StatsProvider getThreadLocalStatsProvider() {
    StatsProvider provider = statsProviderThreadLocal.get();
    if (provider == null) {
      provider = new StatsProvider(sharedReader);
      statsProviderThreadLocal.set(provider);
    }
    return provider;
  }

  public static void main(String[] args) throws Exception {
    String indexPath = null;
    String topicsPath = null;
    String runsOutputFolder = null;

    // Ranking model used as baseline for reranking. Default is LMDirichlet.
    String baselineModel = "LMDirichlet";

    String qrelsPath = null;
    String cacheDir = null;

    String rerankMethod = "none";
    String rfStrategy = "prf";
    String prfModel = "RM3";

    int rerankDepth = 100;
    int e = 20;
    double lambda = 0.0;

    // Grid search parameters
    boolean gridSearch = false;
    String depthsStr = null;
    String eValuesStr = null;
    String lambdasStr = null;

    // Parse arguments
    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--index":
          indexPath = args[++i];
          break;
        case "--topics":
          topicsPath = args[++i];
          break;
        case "--qrels":
          qrelsPath = args[++i];
          break;
        case "--cache_dir":
          cacheDir = args[++i];
          break;
        case "--runsOutputFolder":
          runsOutputFolder = args[++i];
          break;
        case "--baseline_model":
          baselineModel = args[++i];
          break;
        case "--rerank_method":
          rerankMethod = args[++i];
          break;
        case "--rerank_depth":
          rerankDepth = Integer.parseInt(args[++i]);
          break;
        case "--prf_strategy":
          rfStrategy = args[++i];
          break;
        case "--prf_model":
          prfModel = args[++i];
          break;
        case "--lambda":
          lambda = Double.parseDouble(args[++i]);
          break;
        case "-e":
          e = Integer.parseInt(args[++i]);
          break;
        case "--grid_search":
          gridSearch = true;
          break;
        case "--depths":
          depthsStr = args[++i];
          break;
        case "--e_values":
          eValuesStr = args[++i];
          break;
        case "--lambdas":
          lambdasStr = args[++i];
          break;
      }
    }

    // Validate required arguments
    validateRequiredArguments(indexPath, topicsPath, runsOutputFolder);

    // Parse grid search parameters
    int[] depths = { rerankDepth };
    int[] eValues = { e };
    double[] lambdas = { lambda };
    if (gridSearch) {
      if (depthsStr != null) {
        String[] parts = depthsStr.split(",");
        depths = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
          depths[i] = Integer.parseInt(parts[i].trim());
        }
      }
      if (eValuesStr != null) {
        String[] parts = eValuesStr.split(",");
        eValues = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
          eValues[i] = Integer.parseInt(parts[i].trim());
        }
      }
      if (lambdasStr != null) {
        String[] parts = lambdasStr.split(",");
        lambdas = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
          lambdas[i] = Double.parseDouble(parts[i].trim());
        }
      }
    }

    // Open index
    System.out.println("Opening index: " + indexPath);
    IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
    sharedReader = reader; // Store for thread-local StatsProvider creation
    IndexSearcher searcher = new IndexSearcher(reader);
    searcher.setSimilarity(createSimilarity(baselineModel));

    // Parse topics
    System.out.println("Parsing topics: " + topicsPath);
    List<Topic> topics = TRECUtils.parseTRECTopics(topicsPath);

    // Load oracle qrels
    if (rfStrategy.contains("ORACLE")) {
      System.out.println("Parsing qrels for oracle run: " + qrelsPath);
      oracle = loadOracleRelevance(qrelsPath, searcher);
    }

    // Create main thread's stats provider (others will be created on-demand per thread)
    StatsProvider statsProvider = new StatsProvider(searcher.getIndexReader());
    statsProviderThreadLocal.set(statsProvider); // Set for main thread

    // Initialize caches once for all configurations based on strategy
    LLMCache monoT5Cache = null;
    LLMCache vllmCache = null;

    // Initialize cache for PRF strategies
    if (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB") || rerankMethod.equals("monot5")) {
      System.out.println("Initializing MonoT5 cache from " + cacheDir);
      monoT5Cache = new MonoT5Cache(cacheDir);
    } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
      System.out.println("Initializing VLLM cache...");
      vllmCache = new VLLMCache(cacheDir);
    }

    // Grid search loop - optimized to compute RM3 expansion once per (depth, e)
    // combination
    int totalConfigs = depths.length * eValues.length * lambdas.length;
    int currentConfig = 0;
    int skipped = 0;
    if (gridSearch) {
      System.out.println("\n========================================");
      System.out.println("Starting Grid Search PRF Expansion");
      System.out.println("========================================");
      System.out.println("Total configurations: " + totalConfigs);
      System.out.println("RF Strategy: " + rfStrategy);
      System.out.println("Depths: " + Arrays.toString(depths));
      System.out.println("E values: " + Arrays.toString(eValues));
      System.out.println("Lambda values: " + Arrays.toString(lambdas));
      System.out.println("-------------------------------------------------------");
    } else {
      System.out.println("-------------------------------------------------------");
      System.out.println("Rerank: " + rerankMethod);
      System.out.println("RF Strategy: " + rfStrategy);
      System.out.println("Depth: " + Arrays.toString(depths));
      System.out.println("e value: " + Arrays.toString(eValues));
      System.out.println("Lambda value: " + Arrays.toString(lambdas));
      System.out.println("[!]: You can ignore these variables if this run is a baseline");
      System.out.println("-------------------------------------------------------");
    }

    long startTime = System.currentTimeMillis();

    if (rerankMethod.equals("none")) {
      // Baseline run
      runConfigurationBatch(topics,
                            searcher,
                            runsOutputFolder,
                            baselineModel,
                            rerankMethod,
                            rfStrategy,
                            prfModel,
                            rerankDepth,
                            e,
                            new double[] { lambda },
                            monoT5Cache,
                            vllmCache,
                            0,
                            1);
    }

    if (rerankMethod.equals("monot5")) {
      // MonoT5 reranking
      runConfigurationBatch(topics,
                            searcher,
                            runsOutputFolder,
                            baselineModel,
                            rerankMethod,
                            rfStrategy,
                            prfModel,
                            rerankDepth,
                            e,
                            new double[] { lambda },
                            monoT5Cache,
                            vllmCache,
                            0,
                            1);
    }

    if (rerankMethod.equals("prf")) {
      for (int depth : depths) {
        for (int eVal : eValues) {
          runConfigurationBatch(topics,
                                searcher,
                                runsOutputFolder,
                                baselineModel,
                                rerankMethod,
                                rfStrategy,
                                prfModel,
                                depth,
                                eVal,
                                lambdas,
                                monoT5Cache,
                                vllmCache,
                                currentConfig,
                                totalConfigs);

          // Update counters
          currentConfig += lambdas.length;

          // Count how many were actually processed (not skipped)
          for (double lambdaVal : lambdas) {
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, prfModel, depth, lambdaVal, eVal);
            String trecRunPath = runsOutputFolder + "/" + runName;
            File outputFile = new File(trecRunPath);
            if (outputFile.exists()) {
              skipped++;
            }
          }
        }
      }
    }

    // Close caches
    if (monoT5Cache != null) {
      monoT5Cache.close();
    }
    if (vllmCache != null) {
      vllmCache.close();
    }

    reader.close();

    // Summary
    long endTime = System.currentTimeMillis();
    long totalTime = (endTime - startTime) / 1000; // seconds

    System.out.println("\n========================================");
    System.out.println("Search Completed!");
    System.out.println("========================================");
    System.out.println("Total configurations: " + totalConfigs);
    System.out.println("Completed: " + (totalConfigs - skipped));
    System.out.println("Skipped: " + skipped);
    System.out.println("Total time: " + totalTime + "s");
    System.out.println("Average time per config: " + (totalTime / Math.max(1, totalConfigs - skipped)) + "s");
    System.out.println("========================================");
  }

  /**
   * Build the run name based on the rerank method and parameters.
   */
  private static String buildRunName(String baselineModel, String rerankMethod, String rfStrategy, String rfModel,
                                     int depth, double lambda, int e) {

    String baselineStr = baselineModel.equals("LMDirichlet") ? "LMDirichlet-2000" : baselineModel;

    switch (rerankMethod) {
      case "none" -> {
        return String.format("%s_%s", baselineStr, SEARCH_FIELD);
      }
      case "monot5" -> {
        return String.format("%s_%s_rerank-monoT5_topK-%d", baselineStr, SEARCH_FIELD, depth);
      }
      case "prf" -> {
        if (rfStrategy.equals("ORACLE")) {
          return String.format("%s_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-Additive-%.4f_lambda-%.2f_e-%d",
                               baselineStr,
                               SEARCH_FIELD,
                               true,
                               rfStrategy,
                               rfModel,
                               smoothingParameter,
                               lambda,
                               e);

        } else {
          return String.format(
              "%s_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-Additive-%.4f_topK-%d_lambda-%.2f_e-%d",
              baselineStr,
              SEARCH_FIELD,
              true,
              rfStrategy,
              rfModel,
              smoothingParameter,
              depth,
              lambda,
              e);

        }
      }
      default -> throw new IllegalArgumentException("Unknown rerank method: " + rerankMethod);
    }
  }

  // Method to run a batch of configurations for all lambda values with same (depth, e)
  // This optimizes by computing the expanded query once and then varying only lambda
  private static void runConfigurationBatch(List<Topic> topics, IndexSearcher searcher, String trecRunFolder,
                                            String baselineModel, String rerankMethod, String rfStrategy,
                                            String rfModel, int depth, int e, double[] lambdas, LLMCache monoT5Cache,
                                            LLMCache vllmCache, int startConfig, int totalConfigs) throws Exception {

    // First, check which lambda values actually need processing (BEFORE computing)
    List<Double> neededLambdas = new ArrayList<>();
    for (double lambda : lambdas) {
      String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, depth, lambda, e);

      String trecRunPath = trecRunFolder + "/" + runName;
      File outputFile = new File(trecRunPath);

      if (!outputFile.exists()) {
        neededLambdas.add(lambda);
      }
    }

    // If all files exist, skip this entire (depth, e) combination
    if (neededLambdas.isEmpty()) {
      System.out.printf("All %d lambda configs exist for depth=%d, e=%d - skipping RM3 computation%n",
                        lambdas.length,
                        depth,
                        e);
      for (int i = 0; i < lambdas.length; i++) {
        int configNum = startConfig + i + 1;
        System.out.printf("[%d/%d] SKIPPING (exists): lambda=%.2f%n", configNum, totalConfigs, lambdas[i]);
      }
      return;
    }
    System.out.printf("Will process configurations from %d to %d: %n",
                      startConfig + 1,
                      startConfig + neededLambdas.size());
    // Report which lambdas need processing
    System.out.printf("Need to process %d/%d lambda values for depth=%d, e=%d%n",
                      neededLambdas.size(),
                      lambdas.length,
                      depth,
                      e);

    // Initialize output files (create empty files for all needed lambdas to avoid append issues)
    for (double lambda : neededLambdas) {
      String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, depth, lambda, e);
      String trecRunPath = trecRunFolder + "/" + runName;

      // Create empty file (overwrite if exists from incomplete run)
      new FileWriter(trecRunPath, false).close();
    }

    // Process each topic IN PARALLEL for better performance
    // Use thread-safe maps to collect results per lambda
    Map<Double, List<String>> resultsPerLambda = new ConcurrentHashMap<>();
    for (double lambda : neededLambdas) {
      resultsPerLambda.put(lambda, new CopyOnWriteArrayList<>());
    }

    // Determine if we should use parallel processing
    // When caches are empty for LLM strategies, use sequential to avoid overloading
    boolean useParallel = true;
    final String parallelMode; // Final for use in lambda

    if (rerankMethod.equals("monot5") || (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB"))) {
      if ((monoT5Cache != null && monoT5Cache.isEmpty()) || trecRunFolder.contains("test")) {
        useParallel = false;
        parallelMode = "sequential (MonoT5 cache empty or test run)";
        System.out.println("⚠ MonoT5 cache is empty or test run - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
      if ((vllmCache != null && vllmCache.isEmpty()) || trecRunFolder.contains("test")) {
        useParallel = false;
        parallelMode = "sequential (VLLM cache empty or test run)";
        System.out.println("⚠ VLLM cache is empty or test run - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else {
      parallelMode = "parallel";
    }

    // Choose stream based on cache state
    Stream<Topic> topicStream = useParallel ? topics.parallelStream() : topics.stream();

    // Process topics (parallel or sequential based on cache state)
    topicStream.forEach(topic -> {
      try {
        // Get thread-local StatsProvider (creates one if needed for this thread)
        StatsProvider threadStatsProvider = getThreadLocalStatsProvider();

        String queryStr = topic.title;

        QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);
        Query query = parser.parse(QueryParser.escape(queryStr));

        // Get initial results
        TopDocs results = searcher.search(query, 1000);

        // Process based on rerank method
        if (rerankMethod.equals("none")) {
          // Baseline - just use initial results for all lambdas
          for (double lambda : neededLambdas) {
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, depth, lambda, e);

            StringBuilder resultStr = new StringBuilder();
            for (int i = 0; i < Math.min(1000, results.scoreDocs.length); i++) {
              ScoreDoc scoreDoc = results.scoreDocs[i];
              Document doc = searcher.storedFields().document(scoreDoc.doc);
              String docno = doc.get(DOCID_FIELD);
              String tag = (i == 0) ? runName : "--";
              resultStr.append(String.format("%s Q0 %s %d %.6f %s\n", topic.num, docno, i + 1, scoreDoc.score, tag));
            }
            resultsPerLambda.get(lambda).add(resultStr.toString());
          }

        } else if (rerankMethod.equals("monot5")) {
          // MonoT5 reranking
          for (double lambda : neededLambdas) {
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, depth, lambda, e);

            // Rerank with MonoT5
            TopDocs rerankedResults = rerankWithMonoT5(queryStr,
                                                       null,
                                                       Integer.parseInt(topic.num),
                                                       results,
                                                       searcher,
                                                       depth,
                                                       monoT5Cache);

            StringBuilder resultStr = new StringBuilder();
            for (int i = 0; i < Math.min(1000, rerankedResults.scoreDocs.length); i++) {
              ScoreDoc scoreDoc = rerankedResults.scoreDocs[i];
              Document doc = searcher.storedFields().document(scoreDoc.doc);
              String docno = doc.get(DOCID_FIELD);
              String tag = (i == 0) ? runName : "--";
              resultStr.append(String.format("%s Q0 %s %d %.6f %s\n", topic.num, docno, i + 1, scoreDoc.score, tag));
            }
            resultsPerLambda.get(lambda).add(resultStr.toString());
          }

        } else if (rerankMethod.equals("prf")) {
          // PRF with query expansion
          // Compute expanded query weights ONCE for this topic and (depth, e) combination
          // Using thread-local StatsProvider to avoid contention
          TermWeights expandedQueryWeights = queryExpansion(queryStr,
                                                            null,
                                                            Integer.parseInt(topic.num),
                                                            results,
                                                            rfStrategy,
                                                            searcher,
                                                            threadStatsProvider,
                                                            rfModel,
                                                            depth,
                                                            e,
                                                            monoT5Cache,
                                                            vllmCache);

          // Get original query weights
          List<String> processedTerms = new ArrayList<>();
          try (TokenStream tokenStream = analyzer.tokenStream(SEARCH_FIELD, queryStr)) {
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
              processedTerms.add(tokenStream.getAttribute(CharTermAttribute.class).toString());
            }
            tokenStream.end();
          }
          TermWeights originalQueryWeights = TermWeights.fromTerms(processedTerms).scaleToL1Norm();

          // For each lambda value that needs processing, interpolate and search
          for (double lambda : neededLambdas) {
            // Interpolate original query with expanded query
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, depth, lambda, e);

            // Interpolate: lambda controls weight of ORIGINAL query
            // lambda=1.0 → 100% original (no PRF)
            // lambda=0.0 → 100% expanded (full PRF)
            TermWeights finalQuery = TermWeights.interpolate(originalQueryWeights, expandedQueryWeights, lambda);

            // Build query string with weights
            // Escape individual terms, then add boost weights
            StringBuilder queryBuilder = new StringBuilder();
            finalQuery.forEach((term, weight) -> {
              String escapedTerm = QueryParser.escape(term);
              queryBuilder.append(escapedTerm).append("^").append(String.format("%.10f ", weight));
            });

            // Execute search with interpolated query
            Query expandedQuery = parser.parse(queryBuilder.toString());
            TopDocs expandedResults = searcher.search(expandedQuery, 1000);

            // Collect results as strings (to write later in order)
            if (expandedResults != null) {
              StringBuilder resultStr = new StringBuilder();
              for (int i = 0; i < Math.min(1000, expandedResults.scoreDocs.length); i++) {
                ScoreDoc scoreDoc = expandedResults.scoreDocs[i];
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                String docno = doc.get(DOCID_FIELD);
                // Optimization: full runName for rank 1, "--" for the rest to save space
                String tag = (i == 0) ? runName : "--";
                resultStr.append(String.format("%s Q0 %s %d %.6f %s\n", topic.num, docno, i + 1, scoreDoc.score, tag));
              }
              resultsPerLambda.get(lambda).add(resultStr.toString());
            }
          }
        }

        // Progress reporting (synchronized to avoid interleaved output)
        synchronized (System.out) {
          int processed = (int) resultsPerLambda.values().stream().mapToInt(List::size).average().orElse(0);
          if (processed % 10 == 0 || processed == topics.size()) {
            System.out.printf("  Processed %d/%d topics for depth=%d, e=%d (%s)%n",
                              processed,
                              topics.size(),
                              depth,
                              e,
                              parallelMode);
          }
        }

      } catch (Exception e_ex) {
        System.err.println("Error processing topic " + topic.num + ": " + e_ex.getMessage());
        e_ex.printStackTrace();
      }
    });

    // Write all results to files in topic order (sequential, after parallel
    // processing)
    System.out.println("Writing results to files...");
    for (double lambda : neededLambdas) {
      String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, depth, lambda, e);
      String trecRunPath = trecRunFolder + "/" + runName;

      try (BufferedWriter runWriter = new BufferedWriter(new FileWriter(trecRunPath, false))) {
        for (String result : resultsPerLambda.get(lambda)) {
          runWriter.write(result);
        }
      }
    }
    System.out.printf("✓ Completed depth=%d, e=%d with %d lambda values%n", depth, e, neededLambdas.size());

  }

  // Rerank top results using MonoT5
  private static TopDocs rerankWithMonoT5(String queryText, String narrative, int queryId, TopDocs initialResults,
                                          IndexSearcher searcher, int depth, LLMCache cache) throws IOException {

    List<ScoredDoc> scoredDocs = new ArrayList<>();
    int docsToRerank = Math.min(depth, initialResults.scoreDocs.length);

    // Rerank top-depth documents with MonoT5 using score
    for (int i = 0; i < docsToRerank; i++) {
      ScoreDoc sd = initialResults.scoreDocs[i];
      Document doc = searcher.storedFields().document(sd.doc);
      String docText = doc.get(SEARCH_FIELD);

      // Get result from cache or evaluate
      LLMResult result = cache.get(queryId, sd.doc, queryText, narrative, docText);

      // Use the score (prob_true) for reranking
      scoredDocs.add(new ScoredDoc(sd.doc, result.score));
    }

    // Add remaining documents (not reranked) with low scores
    for (int i = docsToRerank; i < initialResults.scoreDocs.length; i++) {
      ScoreDoc sd = initialResults.scoreDocs[i];
      scoredDocs.add(new ScoredDoc(sd.doc, -1000 - i));
    }

    // Sort by MonoT5 scores (descending)
    scoredDocs.sort((a, b) -> Double.compare(b.score, a.score));

    // Convert back to TopDocs
    ScoreDoc[] rerankedScoreDocs = new ScoreDoc[scoredDocs.size()];
    for (int i = 0; i < scoredDocs.size(); i++) {
      ScoredDoc sd = scoredDocs.get(i);
      rerankedScoreDocs[i] = new ScoreDoc(sd.docId, (float) sd.score);
    }

    return new TopDocs(initialResults.totalHits, rerankedScoreDocs);
  }

  private static class ScoredDoc {
    int docId;
    double score;

    ScoredDoc(int docId, double score) {
      this.docId = docId;
      this.score = score;
    }
  }

  private static Map<Integer, Double> filterRelevantDocuments(int queryid, String queryText, String narrative,
                                                              TopDocs results, String rfStrategy, int k,
                                                              IndexSearcher searcher, LLMCache monoT5Cache,
                                                              LLMCache vllmCache) throws IOException {
    switch (rfStrategy) {
      case "PRF":
        // we took the top k documents as relevant returning a map with docid and score:
        Map<Integer, Double> relevantDocs = new HashMap<>();
        for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
          ScoreDoc sd = results.scoreDocs[i];
          relevantDocs.put(sd.doc, (double) sd.score);
        }

        return relevantDocs;
      case "ORACLE":
        // Oracle without limit - collect all oracle-relevant documents
        return filterWithOracle(queryid, results, Integer.MAX_VALUE);
      case "ORACLE-K":
        // Oracle with limit - collect up to k oracle-relevant documents
        return filterWithOracle(queryid, results, k);
      case "MONOT5":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             monoT5Cache,
                             (sd, result) -> (double) sd.score); // Use retrieval score
      case "MONOT5-PROB":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             monoT5Cache,
                             (sd, result) -> result.probTrue); // Use LLM probability
      case "VLLM":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             vllmCache,
                             (sd, result) -> (double) sd.score); // Use retrieval score
      case "VLLM-PROB":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             vllmCache,
                             (sd, result) -> result.probTrue); // Use LLM probability
      default:
        throw new IllegalArgumentException("Unknown RF strategy: " + rfStrategy);
    }
  }

  /**
   * Functional interface for determining the score to use for a document.
   */
  @FunctionalInterface
  private interface ScoreFunction {
    double getScore(ScoreDoc sd, LLMResult result);
  }

  private static Map<Integer, Double> filterWithLLM(int queryid, String queryText, String narrative, TopDocs results,
                                                    int k, IndexSearcher searcher, LLMCache llmCache,
                                                    ScoreFunction scoreFunction) throws IOException {
    Map<Integer, Double> filteredDocs = new HashMap<>();

    for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
      ScoreDoc sd = results.scoreDocs[i];
      Document doc = searcher.storedFields().document(sd.doc);
      String docText = doc.get(SEARCH_FIELD);

      // Get result from LLM cache
      LLMResult result = llmCache.get(queryid, sd.doc, queryText, narrative, docText);

      if (result.isRelevant) {
        // Use the score determined by the scoreFunction
        filteredDocs.put(sd.doc, scoreFunction.getScore(sd, result));
      }
    }

    return filteredDocs;
  }

  private static Map<Integer, Double> filterWithOracle(int queryid, TopDocs results, int maxDocs) {
    Map<Integer, Double> oracleDocs = new HashMap<>();

    if (!oracle.containsKey(queryid)) {
      return oracleDocs; // No oracle judgments for this query
    }

    Set<Integer> relevantDocIds = oracle.get(queryid);
    int count = 0;

    for (ScoreDoc sd : results.scoreDocs) {
      if (relevantDocIds.contains(sd.doc)) {
        oracleDocs.put(sd.doc, (double) sd.score);
        count++;
        if (count >= maxDocs) {
          break; // Stop after collecting maxDocs relevant documents
        }
      }
    }

    return oracleDocs;
  }

  private static TermWeights queryExpansion(String originalQuery, String narrative, int queryId, TopDocs results,
                                            String rfStrategy, IndexSearcher searcher, StatsProvider statsProvider,
                                            String rfModel, int k, int e, LLMCache monoT5Cache,
                                            LLMCache vllmCache) throws IOException {
    Map<Integer, Double> prfDocs = filterRelevantDocuments(queryId,
                                                           originalQuery,
                                                           narrative,
                                                           results,
                                                           rfStrategy,
                                                           k,
                                                           searcher,
                                                           monoT5Cache,
                                                           vllmCache);
    Smoothing smoothing = new AdditiveSmoothing(smoothingParameter, SEARCH_FIELD, statsProvider);

    // Create relevance feedback model based on rfModel parameter
    RelevanceFeedback feedbackModel;
    switch (rfModel.toUpperCase()) {
      case "RM3":
        feedbackModel = new RM3(SEARCH_FIELD, smoothing);
        break;
      case "DMM":
        feedbackModel = new DMM(SEARCH_FIELD, smoothing, DMM_LAMBDA);
        break;
      case "MEDMM":
        feedbackModel = new MEDMM(SEARCH_FIELD, smoothing, MEDMM_LAMBDA, MEDMM_BETA);
        break;
      default:
        throw new IllegalArgumentException("Unknown relevance feedback model: "
                                           + rfModel
                                           + ". Supported models: RM3, DMM, MEDMM");
    }

    // Extract query terms for feedback models that need them
    List<String> queryTerms = extractQueryTerms(originalQuery);

    return feedbackModel.getTermWeights(prfDocs, queryTerms).pruneToSize(e).scaleToL1Norm();
  }

  private static Map<Integer, Set<Integer>> loadOracleRelevance(String pathToQrelsFile,
                                                                IndexSearcher searcher) throws IOException, ParseException {
    // Read TREC qrels file and build map of query_id -> Set<Integer> (docids)
    Map<Integer, Set<Integer>> oracleRelevance = new HashMap<>();
    try (BufferedReader br = new BufferedReader(new FileReader(pathToQrelsFile))) {
      String line;
      while ((line = br.readLine()) != null) {
        String[] parts = line.trim().split("\s+");
        if (parts.length >= 4) {
          Integer queryId = Integer.parseInt(parts[0]);
          String docno = parts[2];
          int relevance = Integer.parseInt(parts[3]);
          if (relevance > 0) { // Only consider relevant documents
            // Find docid from index using docno
            QueryParser docnoParser = new QueryParser(DOCID_FIELD, analyzer);
            Query docnoQuery = docnoParser.parse(QueryParser.escape(docno));

            TopDocs docnoResults = searcher.search(docnoQuery, 1);

            int docid = docnoResults.scoreDocs[0].doc;
            oracleRelevance.computeIfAbsent(queryId, k -> new HashSet<>()).add(docid);
          }
        }
      }
    }
    return oracleRelevance;

  }

  private static List<String> extractQueryTerms(String query) {
    List<String> processedTerms = new ArrayList<>();
    try (TokenStream tokenStream = analyzer.tokenStream(SEARCH_FIELD, query)) {
      tokenStream.reset();
      while (tokenStream.incrementToken()) {
        processedTerms.add(tokenStream.getAttribute(CharTermAttribute.class).toString());
      }
      tokenStream.end();
    } catch (Exception e) {
      throw new RuntimeException("Error extracting query terms", e);
    }
    return processedTerms;
  }

  private static Similarity createSimilarity(String baselineModel) {
    switch (baselineModel.toUpperCase()) {
      case "LMDIRICHLET":
        return new LMDirichletSimilarity(2000f);
      case "BM25":
        return new BM25Similarity();
      default:
        throw new IllegalArgumentException("Unknown baseline model: "
                                           + baselineModel
                                           + ". Supported models: LMDirichlet, BM25");
    }
  }

  private static void validateRequiredArguments(String indexPath, String topicsPath, String trecRunFolder) {
    boolean hasError = false;
    StringBuilder errorMsg = new StringBuilder("Missing required arguments:\n");

    if (indexPath == null) {
      errorMsg.append("  --index <path>         Path to the Lucene index directory\n");
      hasError = true;
    }

    if (topicsPath == null) {
      errorMsg.append("  --topics <path>        Path to the topics file\n");
      hasError = true;
    }

    if (trecRunFolder == null) {
      errorMsg.append("  --runsOutputFolder <path>  Path to output directory for TREC run files\n");
      hasError = true;
    }

    if (hasError) {
      System.err.println("Error: " + errorMsg);
      printUsage();
      System.exit(1);
    }
  }

  private static void printUsage() {
    System.err.println("\nUsage: java -jar prf-llm.jar search [options]");
    System.err.println("\nRequired arguments:");
    System.err.println("  --index <path>                Path to the Lucene index directory");
    System.err.println("  --topics <path>               Path to the topics file");
    System.err.println("  --runsOutputFolder <path>      Path to output directory for TREC run files");

    System.err.println("\nOptional arguments:");
    System.err.println("  --baseline_model <model>      Baseline model: LMDirichlet|BM25 (default: LMDirichlet)");
    System.err.println("  --search_by <field>           Search field (default: content)");
    System.err.println("  --rerank_method <method>      Reranking method: none|monot5|prf (default: none)");
    System.err.println("  --prf_strategy <strategy>     PRF strategy: none|prf|ORACLE|MONOT5|VLLM (default: none)");
    System.err.println("  --rf_model <model>            RF model: RM3|DMM|MEDMM (default: RM3)");
    System.err.println("  --prf_smoothing_model <model> PRF smoothing model (default: Additive)");
    System.err.println("  --prf_smoothing_parameter <p> PRF smoothing parameter (default: 0.1)");
    System.err.println("  --rerank_depth <depth>        Reranking depth (default: 100)");
    System.err.println("  --lambda <lambda>             Lambda for interpolation (default: 0.0)");
    System.err.println("  -e <e>                        Number of expansion terms (default: 20)");

    System.err.println("\nRF model specific arguments:");
    System.err.println("  --dmm_lambda <lambda>         Lambda parameter for DMM (default: 0.5)");
    System.err.println("  --medmm_lambda <lambda>       Lambda parameter for MEDMM (default: 0.5)");
    System.err.println("  --medmm_beta <beta>           Beta parameter for MEDMM (default: 1.0)");

    System.err.println("\nGrid search arguments:");
    System.err.println("  --grid_search                 Enable grid search mode");
    System.err.println("  --depths <depths>             Comma-separated depth values");
    System.err.println("  --e_values <e_values>         Comma-separated e values");
    System.err.println("  --lambdas <lambdas>           Comma-separated lambda values");

    System.err.println("\nConditional arguments (required for some modes):");
    System.err.println("  --qrels <path>                Path to qrels file (required for ORACLE strategy)");
    System.err.println("  --cache_dir <path>            Path to cache directory (required for LLM strategies)");
  }
}
