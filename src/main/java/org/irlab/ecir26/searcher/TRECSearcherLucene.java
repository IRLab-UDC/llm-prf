package org.irlab.ecir26.searcher;

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
import org.irlab.ecir26.searcher.rf.RM3;
import org.irlab.ecir26.searcher.rf.RelevanceFeedback;
import org.irlab.ecir26.searcher.smoothing.AdditiveSmoothing;
import org.irlab.ecir26.searcher.smoothing.Smoothing;
import org.irlab.ecir26.searcher.util.StatsProvider;
import org.irlab.ecir26.searcher.util.TRECUtils;
import org.irlab.ecir26.searcher.util.TRECUtils.Topic;
import org.irlab.ecir26.searcher.util.TermWeights;
import org.irlab.ecir26.searcher.util.scorers.LLMCache;
import org.irlab.ecir26.searcher.util.scorers.LLMResult;
import org.irlab.ecir26.searcher.util.scorers.MonoT5Cache;
import org.irlab.ecir26.searcher.util.scorers.VLLMCache;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class TRECSearcherLucene {

  private static final String SEARCH_FIELD = "content";
  private static final String DOCID_FIELD = "docid";
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
    String qrelsPath = null;
    String cacheDir = null;
    String trecRunFolder = null;
    String searchBy = SEARCH_FIELD;

    String rerankMethod = "none"; // "none", "monot5", "prf"
    String rfStrategy = "none"; // "none", "prf", "ORACLE"
    String rfModel = "RM3";

    String prfSmoothingModel = "Additive";
    double prfSmoothingParameter = 0.1;

    float dirichletMu = 2000f;
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

    // Open index
    System.out.println("Opening index: " + indexPath);
    IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
    sharedReader = reader; // Store for thread-local StatsProvider creation
    IndexSearcher searcher = new IndexSearcher(reader);
    searcher.setSimilarity(new LMDirichletSimilarity(dirichletMu));

    // Parse topics
    System.out.println("Parsing topics: " + topicsPath);
    List<Topic> topics = TRECUtils.parseTRECTopics(topicsPath);

    // Load oracle qrels
    oracle = loadOracleRelevance(qrelsPath, searcher);

    // Create main thread's stats provider (others will be created on-demand per
    // thread)
    StatsProvider statsProvider = new StatsProvider(searcher.getIndexReader());
    statsProviderThreadLocal.set(statsProvider); // Set for main thread

    // Initialize caches once for all configurations based on strategy
    LLMCache monoT5Cache = null;
    LLMCache vllmCache = null;

    // Initialize cache for PRF strategies
    if (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB") || rerankMethod.equals("monot5")) {
      System.out.println("Initializing MonoT5 cache...");
      monoT5Cache = new MonoT5Cache(cacheDir);
    } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
      System.out.println("Initializing VLLM cache...");
      vllmCache = new VLLMCache(cacheDir, searchBy);
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
      runConfigurationBatch(topics,
                            searcher,
                            trecRunFolder,
                            searchBy,
                            rerankMethod,
                            rfStrategy,
                            rfModel,
                            prfSmoothingModel,
                            prfSmoothingParameter,
                            rerankDepth,
                            e,
                            new double[] { lambda },
                            dirichletMu,
                            monoT5Cache,
                            vllmCache,
                            0,
                            1);
    }

    if (rerankMethod.equals("monot5")) {
      // MonoT5 reranking
      runConfigurationBatch(topics,
                            searcher,
                            trecRunFolder,
                            searchBy,
                            rerankMethod,
                            rfStrategy,
                            rfModel,
                            prfSmoothingModel,
                            prfSmoothingParameter,
                            rerankDepth,
                            e,
                            new double[] { lambda },
                            dirichletMu,
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
                                trecRunFolder,
                                searchBy,
                                rerankMethod,
                                rfStrategy,
                                rfModel,
                                prfSmoothingModel,
                                prfSmoothingParameter,
                                depth,
                                eVal,
                                lambdas,
                                dirichletMu,
                                monoT5Cache,
                                vllmCache,
                                currentConfig,
                                totalConfigs);

          // Update counters
          currentConfig += lambdas.length;

          // Count how many were actually processed (not skipped)
          for (double lambdaVal : lambdas) {
            String runName = buildRunName(rerankMethod,
                                          dirichletMu,
                                          searchBy,
                                          rfStrategy,
                                          rfModel,
                                          prfSmoothingModel,
                                          prfSmoothingParameter,
                                          depth,
                                          lambdaVal,
                                          eVal);
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

  /**
   * Build the run name based on the rerank method and parameters.
   */
  private static String buildRunName(String rerankMethod, float dirichletMu, String searchBy, String rfStrategy,
                                     String rfModel, String prfSmoothingModel, double prfSmoothingParameter, int depth,
                                     double lambda, int e) {

    if (rerankMethod.equals("none")) {
      return String.format("LMDirichlet-%.0f_%s", dirichletMu, searchBy);
    } else if (rerankMethod.equals("monot5")) {
      return String.format("LMDirichlet-%.0f_%s_rerank-monoT5_topK-%d", dirichletMu, searchBy, depth);
    } else if (rerankMethod.equals("prf")) {
      if (rfStrategy.equals("ORACLE")) {
        return String.format("LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_lambda-%.2f_e-%d",
                             dirichletMu,
                             searchBy,
                             true,
                             rfStrategy,
                             rfModel,
                             prfSmoothingModel,
                             prfSmoothingParameter,
                             lambda,
                             e);

      } else {
        return String.format(
            "LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_topK-%d_lambda-%.2f_e-%d",
            dirichletMu,
            searchBy,
            true,
            rfStrategy,
            rfModel,
            prfSmoothingModel,
            prfSmoothingParameter,
            depth,
            lambda,
            e);

      }
    } else {
      throw new IllegalArgumentException("Unknown rerank method: " + rerankMethod);
    }
  }

  // Method to run a batch of configurations for all lambda values with same (depth, e)
  // This optimizes by computing the expanded query once and then varying only lambda
  private static void runConfigurationBatch(List<Topic> topics, IndexSearcher searcher, String trecRunFolder,
                                            String searchBy, String rerankMethod, String rfStrategy, String rfModel,
                                            String prfSmoothingModel, double prfSmoothingParameter, int depth, int e,
                                            double[] lambdas, float dirichletMu, LLMCache monoT5Cache,
                                            LLMCache vllmCache, int startConfig, int totalConfigs) throws Exception {

    // First, check which lambda values actually need processing (BEFORE computing RM3!)
    List<Double> neededLambdas = new ArrayList<>();
    for (double lambda : lambdas) {
      String runName = buildRunName(rerankMethod,
                                    dirichletMu,
                                    searchBy,
                                    rfStrategy,
                                    rfModel,
                                    prfSmoothingModel,
                                    prfSmoothingParameter,
                                    depth,
                                    lambda,
                                    e);

      String trecRunPath = trecRunFolder + "/" + runName;
      java.io.File outputFile = new java.io.File(trecRunPath);

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
      String runName = buildRunName(rerankMethod,
                                    dirichletMu,
                                    searchBy,
                                    rfStrategy,
                                    rfModel,
                                    prfSmoothingModel,
                                    prfSmoothingParameter,
                                    depth,
                                    lambda,
                                    e);
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
    java.util.stream.Stream<Topic> topicStream = useParallel ? topics.parallelStream() : topics.stream();

    // Process topics (parallel or sequential based on cache state)
    topicStream.forEach(topic -> {
      try {
        // Get thread-local StatsProvider (creates one if needed for this thread)
        StatsProvider threadStatsProvider = getThreadLocalStatsProvider();

        String queryStr = searchBy.equals("title_plus_description") ? topic.title
                                                                      + " "
                                                                      + topic.description : topic.title;

        QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);
        Query query = parser.parse(QueryParser.escape(queryStr));

        // Get initial results
        TopDocs results = searcher.search(query, 1000);

        // Process based on rerank method
        if (rerankMethod.equals("none")) {
          // Baseline - just use initial results for all lambdas
          for (double lambda : neededLambdas) {
            String runName = buildRunName(rerankMethod,
                                          dirichletMu,
                                          searchBy,
                                          rfStrategy,
                                          rfModel,
                                          prfSmoothingModel,
                                          prfSmoothingParameter,
                                          depth,
                                          lambda,
                                          e);

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
            String runName = buildRunName(rerankMethod,
                                          dirichletMu,
                                          searchBy,
                                          rfStrategy,
                                          rfModel,
                                          prfSmoothingModel,
                                          prfSmoothingParameter,
                                          depth,
                                          lambda,
                                          e);

            // Rerank with MonoT5
            TopDocs rerankedResults = rerankWithMonoT5(queryStr,
                                                       searchBy.equals("title_plus_narrative") ? topic.narrative : null,
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
                                                            searchBy.equals("title_plus_narrative") ? topic.narrative : null,
                                                            Integer.parseInt(topic.num),
                                                            results,
                                                            rfStrategy,
                                                            searcher,
                                                            threadStatsProvider,
                                                            prfSmoothingModel,
                                                            prfSmoothingParameter,
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
            String runName = buildRunName(rerankMethod,
                                          dirichletMu,
                                          searchBy,
                                          rfStrategy,
                                          rfModel,
                                          prfSmoothingModel,
                                          prfSmoothingParameter,
                                          depth,
                                          lambda,
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
      String runName = buildRunName(rerankMethod,
                                    dirichletMu,
                                    searchBy,
                                    rfStrategy,
                                    rfModel,
                                    prfSmoothingModel,
                                    prfSmoothingParameter,
                                    depth,
                                    lambda,
                                    e);
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
                                            String prfSmoothingModel, double prfSmoothingParameter, int k, int e,
                                            LLMCache monoT5Cache, LLMCache vllmCache) throws IOException {
    Map<Integer, Double> prfDocs = filterRelevantDocuments(queryId,
                                                           originalQuery,
                                                           narrative,
                                                           results,
                                                           rfStrategy,
                                                           k,
                                                           searcher,
                                                           monoT5Cache,
                                                           vllmCache);
    Smoothing smoothing = new AdditiveSmoothing(prfSmoothingParameter, SEARCH_FIELD, statsProvider);

    RelevanceFeedback feedbackModel = new RM3(prfSmoothingModel, smoothing);
    return feedbackModel.getTermWeights(prfDocs).pruneToSize(e).scaleToL1Norm();
  }
}
