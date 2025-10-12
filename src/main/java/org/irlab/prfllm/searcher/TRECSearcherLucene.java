package org.irlab.prfllm.searcher;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

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
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.irlab.prfllm.searcher.rf.RM3;
import org.irlab.prfllm.searcher.rf.RelevanceFeedback;
import org.irlab.prfllm.searcher.smoothing.AdditiveSmoothing;
import org.irlab.prfllm.searcher.smoothing.DirichletSmoothing;
import org.irlab.prfllm.searcher.smoothing.Smoothing;
import org.irlab.prfllm.searcher.util.StatsProvider;
import org.irlab.prfllm.searcher.util.TermWeights;
import org.irlab.prfllm.searcher.util.scorers.MonoT5Cache;
import org.irlab.prfllm.searcher.util.scorers.MonoT5Scorer;
import org.irlab.prfllm.searcher.util.scorers.OllamaCache;
import org.irlab.prfllm.searcher.util.scorers.OllamaScorer;
import org.irlab.prfllm.searcher.util.scorers.VLLMCache;
import org.irlab.prfllm.searcher.util.scorers.VLLMScorer;

public class TRECSearcherLucene {

  // Topic class for TREC topics
  private static class Topic {
    String num;
    String title;
    String description;
  }

  private static final String SEARCH_FIELD = "TEXT";

  private static final Analyzer analyzer = new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
  private static Map<Integer, Set<Integer>> oracle;
  
  // ThreadLocal StatsProvider pool - each thread gets its own instance
  private static ThreadLocal<StatsProvider> statsProviderThreadLocal = new ThreadLocal<>();
  private static IndexReader sharedReader; // Shared reader for creating per-thread StatsProviders

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
            QueryParser docnoParser = new QueryParser("DOCNO", analyzer);
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
  
  /**
   * Get thread-local StatsProvider. Creates a new one if this thread doesn't have one yet.
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
    String qrelsPath = null;
    String cacheDir = null;
    String trecRunFolder = null;
    String searchBy = null;

    String rerankMethod = "none"; // "none", "prf", "monot5"
    String rfStrategy = "none"; // "none", "PRF", "ORACLE", "OLLAMA"
    String rfModel = "RM3";

    String prfSmoothingModel = "Additive"; // or "Dirichlet"
    double prfSmoothingParameter = 0.1; // gamma for Additive, mu for Dirichlet

    float dirichletMu = 2000f;
    int rerankDepth = 100;
    int e = 20;
    double lambda = 0.0;

    // Grid search parameters
    boolean gridSearch = false;
    String depthsStr = null;
    String eValuesStr = null;
    String lambdasStr = null;

    String ollamaModel = "llama3.1:8b-instruct-fp16"; // Model to use for Ollama

    // Parse arguments
    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--index_path":
          indexPath = args[++i];
          break;
        case "--topics_path":
          topicsPath = args[++i];
          break;
        case "--qrels_path":
          qrelsPath = args[++i];
          break;
        case "--cache_dir":
          cacheDir = args[++i];
          break;
        case "--trec_run_folder":
          trecRunFolder = args[++i];
          break;
        case "--search_by":
          searchBy = args[++i];
          break;
        case "--rerank_method":
          rerankMethod = args[++i];
          break;
        case "--prf_strategy":
          rfStrategy = args[++i];
          break;
        case "--rf_model":
          rfModel = args[++i];
          break;
        case "--prf_smoothing_model":
          prfSmoothingModel = args[++i];
          break;
        case "--prf_smoothing_parameter":
          prfSmoothingParameter = Double.parseDouble(args[++i]);
          break;
        case "--mu":
          dirichletMu = Float.parseFloat(args[++i]);
          break;
        case "--rerank_depth":
          rerankDepth = Integer.parseInt(args[++i]);
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
        case "--ollama_model":
          ollamaModel = args[++i];
          break;
      }
    }

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

    // Open index once
    System.out.println("Opening index: " + indexPath);
    IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
    sharedReader = reader; // Store for thread-local StatsProvider creation
    IndexSearcher searcher = new IndexSearcher(reader);
    oracle = loadOracleRelevance(qrelsPath, searcher);
    searcher.setSimilarity(new LMDirichletSimilarity(dirichletMu));

    // Configure Ollama model if using OLLAMA strategy
    if (rfStrategy.equals("OLLAMA")) {
      OllamaScorer.setModel(ollamaModel);
    }

    // Parse topics once
    System.out.println("Parsing topics: " + topicsPath);
    List<Topic> topics = parseTRECTopics(topicsPath);

    // Create main thread's stats provider (others will be created on-demand per thread)
    StatsProvider statsProvider = new StatsProvider(searcher.getIndexReader());
    statsProviderThreadLocal.set(statsProvider); // Set for main thread

    // Initialize caches once for all configurations based on strategy
    MonoT5Cache monoT5Cache = null;
    OllamaCache ollamaCache = null;
    VLLMCache vllmCache = null;

    // Initialize cache for PRF strategies
    if (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB")) {
      System.out.println("Initializing MonoT5 cache...");
      monoT5Cache = new MonoT5Cache(cacheDir);
    } else if (rfStrategy.equals("OLLAMA")) {
      System.out.println("Initializing Ollama cache...");
      ollamaCache = new OllamaCache(cacheDir, ollamaModel);
    } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
      System.out.println("Initializing VLLM cache...");
      vllmCache = new VLLMCache(cacheDir);
    }

    // Also initialize MonoT5 cache if using monot5 reranking
    if (rerankMethod.equals("monot5") && monoT5Cache == null) {
      System.out.println("Initializing MonoT5 cache for reranking from " + cacheDir);
      monoT5Cache = new MonoT5Cache(cacheDir);
    }

    // Grid search loop - optimized to compute RM3 expansion once per (depth, e)
    // combination
    int totalConfigs = depths.length * eValues.length * lambdas.length;
    int currentConfig = 0;
    int skipped = 0;
    if (gridSearch) {
      System.out.println("\n========================================");
      System.out.println("Starting Grid Search RM3 Expansion");
      System.out.println("========================================");
      System.out.println("Total configurations: " + totalConfigs);
      System.out.println("RF Strategy: " + rfStrategy);
      System.out.println("Depths: " + java.util.Arrays.toString(depths));
      System.out.println("E values: " + java.util.Arrays.toString(eValues));
      System.out.println("Lambda values: " + java.util.Arrays.toString(lambdas));
      System.out.println("-------------------------------------------------------");
    } else {
      System.out.println("-------------------------------------------------------");
      System.out.println("Rerank: " + rerankMethod);
      System.out.println("Strategy: " + rfStrategy);
      System.out.println("Depths: " + java.util.Arrays.toString(depths));
      System.out.println("E values: " + java.util.Arrays.toString(eValues));
      System.out.println("Lambda values: " + java.util.Arrays.toString(lambdas));
      System.out.println("-------------------------------------------------------");
    }

    long startTime = System.currentTimeMillis();

    if (rerankMethod.equals("none")) {
      // Baseline run
      String runName = String.format("LMDirichlet-%.0f_%s", dirichletMu, searchBy);
      runConfiguration(topics, searcher, statsProvider, trecRunFolder + "/" + runName, runName, searchBy, rerankMethod,
          rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter, rerankDepth, e, lambda,
          ollamaModel, monoT5Cache, ollamaCache, vllmCache);
    }

    if (rerankMethod.equals("monot5")) {
      String runName = String.format("LMDirichlet-%.0f_%s_rerank-monoT5_topK-%d", dirichletMu, searchBy, rerankDepth);
      runConfiguration(topics, searcher, statsProvider, trecRunFolder + "/" + runName, runName, searchBy, rerankMethod,
          rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter, rerankDepth, e, lambda,
          ollamaModel, monoT5Cache, ollamaCache, vllmCache);
    }

    if (rerankMethod.equals("prf")) {
      for (int depth : depths) {
        for (int eVal : eValues) {
          runConfigurationBatch(topics, searcher, statsProvider, trecRunFolder, searchBy, rerankMethod, rfStrategy,
              rfModel, prfSmoothingModel, prfSmoothingParameter, depth, eVal, lambdas, ollamaModel,
              dirichletMu, monoT5Cache, ollamaCache, vllmCache, currentConfig, totalConfigs, skipped);

          // Update counters
          currentConfig += lambdas.length;

          // Count how many were actually processed (not skipped)
          for (double lambdaVal : lambdas) {
            String runName = String.format(
                "LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_topK-%d_lambda-%.2f_e-%d",
                dirichletMu, searchBy, true, rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter, depth,
                lambdaVal, eVal);
            String trecRunPath = trecRunFolder + "/" + runName;
            java.io.File outputFile = new java.io.File(trecRunPath);
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
    if (ollamaCache != null) {
      ollamaCache.close();
    }
    if (vllmCache != null) {
      vllmCache.close();
    }

    reader.close();

    // Summary
    long endTime = System.currentTimeMillis();
    long totalTime = (endTime - startTime) / 1000; // seconds

    System.out.println("\n========================================");
    System.out.println("Grid Search Completed!");
    System.out.println("========================================");
    System.out.println("Total configurations: " + totalConfigs);
    System.out.println("Completed: " + (totalConfigs - skipped));
    System.out.println("Skipped: " + skipped);
    System.out.println("Total time: " + totalTime + "s");
    System.out.println("Average time per config: " + (totalTime / Math.max(1, totalConfigs - skipped)) + "s");
    System.out.println("========================================");
  }

  // New method to run a batch of configurations for all lambda values with same
  // (depth, e)
  // This optimizes by computing the expanded query once and then varying only
  // lambda
  private static void runConfigurationBatch(List<Topic> topics, IndexSearcher searcher, StatsProvider statsProvider,
      String trecRunFolder, String searchBy, String rerankMethod,
      String rfStrategy, String rfModel, String prfSmoothingModel,
      double prfSmoothingParameter, int depth, int e, double[] lambdas,
      String ollamaModel, float dirichletMu, MonoT5Cache monoT5Cache,
      OllamaCache ollamaCache, VLLMCache vllmCache, int startConfig,
      int totalConfigs, int currentSkipped) throws Exception {

    // First, check which lambda values actually need processing (BEFORE computing
    // RM3!)
    List<Double> neededLambdas = new ArrayList<>();
    for (double lambda : lambdas) {
      String runName = String.format(
          "LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_topK-%d_lambda-%.2f_e-%d",
          dirichletMu, searchBy, true, rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter, depth, lambda, e);

      String trecRunPath = trecRunFolder + "/" + runName;
      java.io.File outputFile = new java.io.File(trecRunPath);

      if (!outputFile.exists()) {
        neededLambdas.add(lambda);
      }
    }

    // If all files exist, skip this entire (depth, e) combination
    if (neededLambdas.isEmpty()) {
      System.out.println(String.format("  All %d lambda configs exist for depth=%d, e=%d - skipping RM3 computation",
          lambdas.length, depth, e));
      for (int i = 0; i < lambdas.length; i++) {
        int configNum = startConfig + i + 1;
        System.out.println(
            String.format("  [%d/%d] SKIPPING (exists): lambda=%.2f", configNum, totalConfigs, lambdas[i]));
      }
      return; // Early exit - don't compute anything
    }
    System.out.println(String.format("Will process configurations from %d to %d: ", startConfig + 1,
        startConfig + neededLambdas.size() - currentSkipped));
    // Report which lambdas need processing
    System.out.println(String.format("  Need to process %d/%d lambda values for depth=%d, e=%d", neededLambdas.size(),
        lambdas.length, depth, e));

    // Initialize output files (create empty files for all needed lambdas to avoid append issues)
    for (double lambda : neededLambdas) {
      String runName = String.format(
          "LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_topK-%d_lambda-%.2f_e-%d",
          dirichletMu, searchBy, true, rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter, depth, lambda, e);
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
    
    if (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB")) {
      if (monoT5Cache != null && monoT5Cache.isEmpty()) {
        useParallel = false;
        parallelMode = "sequential (MonoT5 cache empty)";
        System.out.println("⚠ MonoT5 cache is empty - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
      if (vllmCache != null && vllmCache.isEmpty()) {
        useParallel = false;
        parallelMode = "sequential (VLLM cache empty)";
        System.out.println("⚠ VLLM cache is empty - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else {
      parallelMode = "parallel";
    }
    
    // Choose stream based on cache state
    java.util.stream.Stream<Topic> topicStream = useParallel ? topics.parallelStream() : topics.stream();
    
    // Process topics (parallel or sequential based on cache state)
    topicStream.forEach(topic -> {
      try {
        // Get thread-local StatsProvider (creates one if needed for this thread)
        StatsProvider threadStatsProvider = getThreadLocalStatsProvider();
        
        String queryStr = searchBy.equals("title_plus_description") ? topic.title + " " + topic.description : topic.title;

        QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);
        Query query = parser.parse(QueryParser.escape(queryStr));

        // Get initial results
        TopDocs results = searcher.search(query, 1000);

        // Compute expanded query weights ONCE for this topic and (depth, e) combination
        // Using thread-local StatsProvider to avoid contention
        TermWeights expandedQueryWeights = queryExpansion(queryStr, Integer.parseInt(topic.num), results, rfModel,
            rfStrategy, searcher, threadStatsProvider, prfSmoothingModel,
            prfSmoothingParameter, depth, e, ollamaModel, monoT5Cache,
            ollamaCache, vllmCache);

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
          String runName = String.format(
              "LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_topK-%d_lambda-%.2f_e-%d",
              dirichletMu, searchBy, true, rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter, depth, lambda,
              e);

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
          
          // Execute search with interpolated query (DON'T escape - already has Lucene syntax!)
          Query expandedQuery = parser.parse(queryBuilder.toString());
          TopDocs expandedResults = searcher.search(expandedQuery, 1000);

          // Collect results as strings (to write later in order)
          if (expandedResults != null) {
            StringBuilder resultStr = new StringBuilder();
            for (int i = 0; i < Math.min(1000, expandedResults.scoreDocs.length); i++) {
              ScoreDoc scoreDoc = expandedResults.scoreDocs[i];
              Document doc = searcher.storedFields().document(scoreDoc.doc);
              String docno = doc.get("DOCNO");
              // Optimization: full runName for rank 1, "--" for the rest to save space
              String tag = (i == 0) ? runName : "--";
              resultStr.append(String.format("%s Q0 %s %d %.6f %s\n", 
                  topic.num, docno, i + 1, scoreDoc.score, tag));
            }
            resultsPerLambda.get(lambda).add(resultStr.toString());
          }
        }
        
        // Progress reporting (synchronized to avoid interleaved output)
        synchronized (System.out) {
          int processed = (int) resultsPerLambda.values().stream()
              .mapToInt(List::size)
              .average()
              .orElse(0);
          if (processed % 10 == 0 || processed == topics.size()) {
            System.out.println(String.format("  Processed %d/%d topics for depth=%d, e=%d (%s)", 
                processed, topics.size(), depth, e, parallelMode));
          }
        }
        
      } catch (Exception e_ex) {
        System.err.println("Error processing topic " + topic.num + ": " + e_ex.getMessage());
        e_ex.printStackTrace();
      }
    });

    // Write all results to files in topic order (sequential, after parallel processing)
    System.out.println("Writing results to files...");
    for (double lambda : neededLambdas) {
      String runName = String.format(
          "LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_topK-%d_lambda-%.2f_e-%d",
          dirichletMu, searchBy, true, rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter, depth, lambda, e);
      String trecRunPath = trecRunFolder + "/" + runName;
      
      try (BufferedWriter runWriter = new BufferedWriter(new FileWriter(trecRunPath, false))) {
        for (String result : resultsPerLambda.get(lambda)) {
          runWriter.write(result);
        }
      }
    }
    System.out.println(String.format("✓ Completed depth=%d, e=%d with %d lambda values", depth, e, neededLambdas.size()));

  }

  // Original method kept for single configuration runs (backward compatibility)
  private static void runConfiguration(List<Topic> topics, IndexSearcher searcher, StatsProvider statsProvider,
      String trecRunPath, String runName, String searchBy, String rerankMethod,
      String rfStrategy, String rfModel, String prfSmoothingModel,
      double prfSmoothingParameter, int rerankDepth, int e, double lambda,
      String ollamaModel, MonoT5Cache monoT5Cache, OllamaCache ollamaCache,
      VLLMCache vllmCache) throws Exception {

    BufferedWriter runWriter = new BufferedWriter(new FileWriter(trecRunPath));

    for (Topic topic : topics) {
      String queryStr = searchBy.equals("title_plus_description") ? topic.title + " " + topic.description : topic.title;

      QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);
      Query query = parser.parse(QueryParser.escape(queryStr));

      TopDocs results = searcher.search(query, 1000);

      if (rerankMethod.equals("monot5")) {
        // Direct reranking with MonoT5 (no query expansion)
        TopDocs rerankedResults = rerankWithMonoT5(queryStr, Integer.parseInt(topic.num), results, searcher,
            rerankDepth, monoT5Cache);
        writeTrecRun(runWriter, topic.num, rerankedResults, searcher, runName);
      } else if (rerankMethod.equals("prf")) {
        // PRF with query expansion
        TermWeights expandedQueryWeights = queryExpansion(queryStr, Integer.parseInt(topic.num), results, rfModel,
            rfStrategy, searcher, statsProvider, prfSmoothingModel,
            prfSmoothingParameter, rerankDepth, e, ollamaModel,
            monoT5Cache, ollamaCache, vllmCache);

        // Second round with expanded query
        // Escape individual terms, then add boost weights
        StringBuilder expandedQueryBuilder = new StringBuilder();
        expandedQueryWeights.forEach((term, weight) -> {
          String escapedTerm = QueryParser.escape(term);
          expandedQueryBuilder.append(escapedTerm).append("^").append(String.format("%.10f ", weight));
        });
        System.out.println("Expanded query for topic " + topic.num + ": " + expandedQueryBuilder.toString());
        // Parse query with boost syntax (DON'T escape - already has Lucene syntax!)
        Query expandedQuery = parser.parse(expandedQueryBuilder.toString());
        TopDocs expandedResults = searcher.search(expandedQuery, 1000);

        writeTrecRun(runWriter, topic.num, expandedResults, searcher, runName);
      } else {
        // No PRF, just write original results
        writeTrecRun(runWriter, topic.num, results, searcher, runName);
      }
    }

    runWriter.close();
  }

  // Parse TREC topics file (simple version)
  private static List<Topic> parseTRECTopics(String topicsPath) throws IOException {
    List<Topic> topics = new ArrayList<>();
    BufferedReader br = new BufferedReader(new FileReader(topicsPath));
    String line;
    Topic topic = null;
    while ((line = br.readLine()) != null) {
      if (line.trim().startsWith("<num>")) {
        topic = new Topic();
        topic.num = line.replaceAll("[^0-9]", "");
      } else if (line.trim().startsWith("<title>")) {
        // We also remove the Topic: prefix if present
        if (topic != null)
          topic.title = line.replace("<title>", "").replace("Topic:", "").trim();

      } else if (line.trim().startsWith("<desc>")) {
        if (topic != null)
          topic.description = br.readLine().replace("Description:", "").trim();
      } else if (line.trim().startsWith("</top>")) {
        if (topic != null)
          topics.add(topic);
      }
    }
    br.close();
    return topics;
  }

  // Write TREC run file
  private static void writeTrecRun(BufferedWriter writer, String qid, TopDocs results, IndexSearcher searcher,
      String runTag) throws IOException {
    int rank = 1;
    for (ScoreDoc sd : results.scoreDocs) {
      Document doc = searcher.storedFields().document(sd.doc);
      String docno = doc.get("DOCNO");
      // Write full runTag for rank 1, use "--" for the rest to save space
      String tag = (rank == 1) ? runTag : "--";
      writer.write(String.format("%s Q0 %s %d %f %s\n", qid, docno, rank, sd.score, tag));
      rank++;
    }
  }

  // Rerank top results using MonoT5
  private static TopDocs rerankWithMonoT5(String queryText, int queryId, TopDocs initialResults, IndexSearcher searcher,
      int depth, MonoT5Cache cache) throws IOException {

    List<ScoredDoc> scoredDocs = new ArrayList<>();
    int docsToRerank = Math.min(depth, initialResults.scoreDocs.length);

    // Rerank top-depth documents with MonoT5 using score
    for (int i = 0; i < docsToRerank; i++) {
      ScoreDoc sd = initialResults.scoreDocs[i];
      Document doc = searcher.storedFields().document(sd.doc);
      String docText = doc.get("TEXT");

      // Get result from cache or evaluate
      MonoT5Scorer.MonoT5Result result = cache.get(queryId, sd.doc, queryText, docText);

      // Use the score (prob_true) for reranking
      scoredDocs.add(new ScoredDoc(sd.doc, result.score));
    }

    // Add remaining documents (not reranked) with very low scores
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
  } // Helper class for reranking

  private static class ScoredDoc {
    int docId;
    double score;

    ScoredDoc(int docId, double score) {
      this.docId = docId;
      this.score = score;
    }
  }

  private static Map<Integer, Double> filterRelevantDocuments(int queryid, String queryText, TopDocs results, 
      String rfStrategy, int k, IndexSearcher searcher,
      String ollamaModel, MonoT5Cache monoT5Cache,
      OllamaCache ollamaCache,
      VLLMCache vllmCache) throws IOException {
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
        Map<Integer, Double> oracleDocs = new HashMap<>();
        for (ScoreDoc sd : results.scoreDocs) {
          if (oracle.containsKey(queryid)) {
            if (oracle.get(queryid).contains(sd.doc)) {
              oracleDocs.put(sd.doc, (double) sd.score);
            }
          }
        }
        return oracleDocs;
      case "ORACLE-K":
        Map<Integer, Double> oracleKDocs = new HashMap<>();
        int count = 0;
        for (ScoreDoc sd : results.scoreDocs) {
          if (oracle.containsKey(queryid)) {
            if (oracle.get(queryid).contains(sd.doc)) {
              oracleKDocs.put(sd.doc, (double) sd.score);
              count++;
              if (count >= k) {
                break; // Stop after collecting k relevant documents
              }
            }
          }
        }
        return oracleKDocs;
      case "MONOT5":
        // Use MonoT5 model as a binary filter: iterate through documents in rank order
        // and collect the first k documents that MonoT5 classifies as relevant.
        Map<Integer, Double> monoT5Docs = new HashMap<>();

        for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
          ScoreDoc sd = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(sd.doc);
          String docText = doc.get("TEXT");

          // Get result from unified cache
          MonoT5Scorer.MonoT5Result result = monoT5Cache.get(queryid, sd.doc, queryText, docText);

          if (result.isRelevant) {
            // Use the original retrieval score for weighting in RM3
            monoT5Docs.put(sd.doc, (double) sd.score);
          }
        }

        return monoT5Docs;
      case "MONOT5-PROB":
        // Use MonoT5 model as a binary filter: iterate through documents in rank order
        // and collect the first k documents that MonoT5 classifies as relevant.
        Map<Integer, Double> monoT5ProbDocs = new HashMap<>();

        for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
          ScoreDoc sd = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(sd.doc);
          String docText = doc.get("TEXT");

          // Get result from unified cache
          MonoT5Scorer.MonoT5Result result = monoT5Cache.get(queryid, sd.doc, queryText, docText);

          if (result.isRelevant) {
            // Use the original retrieval score for weighting in RM3
            monoT5ProbDocs.put(sd.doc, result.probTrue);
          }
        }

        return monoT5ProbDocs;
      case "OLLAMA":
        // Use Ollama LLM as a binary filter: iterate through documents in rank order
        // and collect the first k documents that Ollama classifies as relevant.
        Map<Integer, Double> ollamaDocs = new HashMap<>();

        for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
          ScoreDoc sd = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(sd.doc);
          String docText = doc.get("TEXT");

          // Get result from Ollama cache
          OllamaScorer.OllamaResult result = ollamaCache.get(queryid, sd.doc, queryText, docText);

          if (result.isRelevant) {
            // Use the original retrieval score for weighting in RM3
            ollamaDocs.put(sd.doc, (double) sd.score);
          }
        }

        return ollamaDocs;
      case "VLLM":
        // Use VLLM as a binary filter: iterate through documents in rank order
        // and collect the first k documents that VLLM classifies as relevant.
        Map<Integer, Double> vllmDocs = new HashMap<>();

        for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
          ScoreDoc sd = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(sd.doc);
          String docText = doc.get("TEXT");

          // Get result from VLLM cache
          VLLMScorer.VLLMResult result = vllmCache.get(queryid, sd.doc, queryText, docText);

          if (result.isRelevant) {
            // Use the original retrieval score for weighting in RM3
            vllmDocs.put(sd.doc, (double) sd.score);
          }
        }

        return vllmDocs;
      case "VLLM-PROB":
        // Use VLLM with probability scores: iterate through documents in rank order
        // and collect the first k documents that VLLM classifies as relevant,
        // using probTrue as the document weight.
        Map<Integer, Double> vllmProbDocs = new HashMap<>();

        for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
          ScoreDoc sd = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(sd.doc);
          String docText = doc.get("TEXT");

          // Get result from VLLM cache
          VLLMScorer.VLLMResult result = vllmCache.get(queryid, sd.doc, queryText, docText);

          if (result.isRelevant) {
            // Use the probability of true as weight
            vllmProbDocs.put(sd.doc, result.probTrue);
          }
        }

        return vllmProbDocs;
      default:
        throw new IllegalArgumentException("Unknown RF strategy: " + rfStrategy);
    }
  }

  private static RelevanceFeedback getRelevanceFeedbackModel(String modelName, String field, Smoothing smoothing) {
    if (modelName.equals("RM3")) {
      return new RM3(field, smoothing);
    }
    throw new IllegalArgumentException("Unknown RF model: " + modelName);
  }

  private static Smoothing geSmoothing(String modelName, double parameter, String field, StatsProvider statsProvider) {
    return switch (modelName) {
      case "Dirichlet" -> new DirichletSmoothing(parameter, field, statsProvider);
      case "Additive" -> new AdditiveSmoothing(parameter, field, statsProvider);
      default -> throw new IllegalArgumentException("Unknown smoothing model: " + modelName);
    };
  }

  private static TermWeights queryExpansion(String originalQuery, int queryId, TopDocs results, String rfModel,
      String rfStrategy, IndexSearcher searcher, StatsProvider statsProvider,
      String prfSmoothingModel, double prfSmoothingParameter, int k, int e,
      String ollamaModel, MonoT5Cache monoT5Cache, OllamaCache ollamaCache,
      VLLMCache vllmCache) throws IOException {

    Map<Integer, Double> prfDocs = filterRelevantDocuments(queryId, originalQuery, results, rfStrategy, k, searcher,
        ollamaModel, monoT5Cache, ollamaCache, vllmCache);
    Smoothing smoothing = geSmoothing(prfSmoothingModel, prfSmoothingParameter, SEARCH_FIELD, statsProvider);
    RelevanceFeedback feedbackModel = getRelevanceFeedbackModel(rfModel, prfSmoothingModel, smoothing);
    return feedbackModel.getTermWeights(prfDocs).pruneToSize(e).scaleToL1Norm();
  }
}
