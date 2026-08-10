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
import org.irlab.llmprf.searcher.util.scorers.TermsProvider;
import org.irlab.llmprf.searcher.util.scorers.VLLMSpansCache;
import org.irlab.llmprf.searcher.util.scorers.VLLMJudgeSpansCache;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Stream;
import java.util.stream.Collectors;

public class TRECSearcherLucene {

  private static final String SEARCH_FIELD = "content";
  private static final String DOCID_FIELD = "docid";
  private static final double smoothingParameter = 0.1d;
  private static final double DMM_LAMBDA = 0.03;
  private static final double MEDMM_LAMBDA = 0.1d;
  private static final double MEDMM_BETA = 1.2d;

  private static final Analyzer analyzer = new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
  private static Map<Integer, Set<Integer>> oracle;

  private static ThreadLocal<StatsProvider> statsProviderThreadLocal = new ThreadLocal<>();
  private static IndexReader sharedReader;

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

    String baselineModel = "LMDirichlet";

    String qrelsPath = null;
    String cacheDir = null;

    String rerankMethod = "none";
    String rfStrategy = "prf";
    String prfModel = "RM3";
    String termFilter = "none";

    int rerankDepth = 100;
    int e = 20;
    double lambda = 0.0;

    boolean gridSearch = false;
    String depthsStr = null;
    String eValuesStr = null;
    String lambdasStr = null;

    String topicId = null;
    int topN = 10;
    String compareSpec = null;

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
        case "--term_filter":
          termFilter = args[++i];
          break;
        case "--topic_id":
          topicId = args[++i];
          break;
        case "--top_n":
          topN = Integer.parseInt(args[++i]);
          break;
        case "--compare":
          compareSpec = args[++i];
          break;
      }
    }

    validateRequiredArguments(indexPath, topicsPath, runsOutputFolder);

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

    System.out.println("Opening index: " + indexPath);
    IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
    sharedReader = reader;
    IndexSearcher searcher = new IndexSearcher(reader);
    searcher.setSimilarity(createSimilarity(baselineModel));

    System.out.println("Parsing topics: " + topicsPath);
    List<Topic> topics = TRECUtils.parseTRECTopics(topicsPath);

    if (rfStrategy.contains("ORACLE")) {
      System.out.println("Parsing qrels for oracle run: " + qrelsPath);
      oracle = loadOracleRelevance(qrelsPath, searcher);
    }

    StatsProvider statsProvider = new StatsProvider(searcher.getIndexReader());
    statsProviderThreadLocal.set(statsProvider);

    LLMCache monoT5Cache = null;
    LLMCache vllmCache = null;
    LLMCache vllmNarrCache = null;
    TermsProvider termsProvider = null;

    if (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB") || rerankMethod.equals("monot5")) {
      System.out.println("Initializing MonoT5 cache from " + cacheDir);
      monoT5Cache = new MonoT5Cache(cacheDir);
    } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
      System.out.println("Initializing VLLM cache...");
      vllmCache = new VLLMCache(cacheDir);
    } else if (rfStrategy.equals("VLLM-NARR") || rfStrategy.equals("VLLM-NARR-PROB")) {
      System.out.println("Initializing VLLM narrative cache...");
      vllmNarrCache = new VLLMCache(cacheDir, "vllm_narr_cache.tsv");
    } else if (rfStrategy.equals("VLLM-JUDGESPANS")) {
      System.out.println("Initializing VLLM judge+spans cache...");
      VLLMJudgeSpansCache judgeSpansCache = new VLLMJudgeSpansCache(cacheDir, false);
      vllmCache = judgeSpansCache.asLLMCache();
      if (termFilter.equals("vllmjudgespans")) {
        termsProvider = judgeSpansCache.asTermsProvider();
      }
    } else if (rfStrategy.equals("VLLM-NARR-JUDGESPANS")) {
      System.out.println("Initializing VLLM judge+spans cache (narrative)...");
      VLLMJudgeSpansCache judgeSpansCache = new VLLMJudgeSpansCache(cacheDir, true);
      vllmNarrCache = judgeSpansCache.asLLMCache();
      if (termFilter.equals("vllmjudgespans")) {
        termsProvider = judgeSpansCache.asTermsProvider();
      }
    }
    if (termFilter.equals("vllmspans2")) {
      System.out.println("Initializing VLLM spans cache...");
      termsProvider = new VLLMSpansCache(cacheDir);
    } else if (termFilter.equals("vllmspans2-nonarr")) {
      System.out.println("Initializing VLLM spans cache (no narrative)...");
      termsProvider = new VLLMSpansCache(cacheDir, false);
    }

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
      runConfigurationBatch(topics,
                            searcher,
                            runsOutputFolder,
                            baselineModel,
                            rerankMethod,
                            rfStrategy,
                            prfModel,
                            termFilter,
                            rerankDepth,
                            e,
                            new double[] { lambda },
                            monoT5Cache,
                            vllmCache,
                            vllmNarrCache,
                            termsProvider,
                            0,
                            1);
    }

    if (rerankMethod.equals("monot5")) {
      runConfigurationBatch(topics,
                            searcher,
                            runsOutputFolder,
                            baselineModel,
                            rerankMethod,
                            rfStrategy,
                            prfModel,
                            termFilter,
                            rerankDepth,
                            e,
                            new double[] { lambda },
                            monoT5Cache,
                            vllmCache,
                            vllmNarrCache,
                            termsProvider,
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
                                termFilter,
                                depth,
                                eVal,
                                lambdas,
                                monoT5Cache,
                                vllmCache,
                                vllmNarrCache,
                                termsProvider,
                                currentConfig,
                                totalConfigs);

          for (double lambdaVal : lambdas) {
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, prfModel, termFilter, depth, lambdaVal, eVal);
            File outputFile = new File(runsOutputFolder + "/" + runName);
            if (outputFile.exists()) {
              skipped++;
            }
          }

          currentConfig += lambdas.length;
        }
      }
    }

    if (rerankMethod.equals("dump_terms")) {
      if (topicId == null || compareSpec == null) {
        throw new IllegalArgumentException("--rerank_method dump_terms requires --topic_id and --compare");
      }
      runDumpTerms(topics, searcher, statsProvider, topicId, compareSpec, topN, cacheDir, qrelsPath);
    }

    if (monoT5Cache != null) {
      monoT5Cache.close();
    }
    if (vllmCache != null) {
      vllmCache.close();
    }
    if (vllmNarrCache != null) {
      vllmNarrCache.close();
    }
    if (termsProvider != null) {
      termsProvider.close();
    }

    reader.close();

    long endTime = System.currentTimeMillis();
    long totalTime = (endTime - startTime) / 1000;

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

  private static String buildRunName(String baselineModel, String rerankMethod, String rfStrategy, String rfModel,
                                     String termFilter, int depth, double lambda, int e) {

    String baselineStr = baselineModel.equals("LMDirichlet") ? "LMDirichlet-2000" : baselineModel;
    String termFilterSuffix = termFilter.equals("none") ? "" : "_termFilter-" + termFilter.toUpperCase();

    switch (rerankMethod) {
      case "none" -> {
        return String.format("%s_%s", baselineStr, SEARCH_FIELD);
      }
      case "monot5" -> {
        return String.format("%s_%s_rerank-monoT5_topK-%d", baselineStr, SEARCH_FIELD, depth);
      }
      case "prf" -> {
        return String.format(
            "%s_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-Additive-%.4f_topK-%d_lambda-%.2f_e-%d%s",
            baselineStr, SEARCH_FIELD, true, rfStrategy, rfModel,
            smoothingParameter, depth, lambda, e, termFilterSuffix);
      }
      default -> throw new IllegalArgumentException("Unknown rerank method: " + rerankMethod);
    }
  }

  private static void runConfigurationBatch(List<Topic> topics, IndexSearcher searcher, String trecRunFolder,
                                            String baselineModel, String rerankMethod, String rfStrategy,
                                            String rfModel, String termFilter, int depth, int e, double[] lambdas,
                                            LLMCache monoT5Cache, LLMCache vllmCache, LLMCache vllmNarrCache,
                                            TermsProvider termsProvider, int startConfig, int totalConfigs)
      throws Exception {

    List<Double> neededLambdas = new ArrayList<>();
    for (double lambda : lambdas) {
      String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, termFilter, depth, lambda, e);

      String trecRunPath = trecRunFolder + "/" + runName;
      File outputFile = new File(trecRunPath);

      if (!outputFile.exists()) {
        neededLambdas.add(lambda);
      }
    }

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
    System.out.printf("Need to process %d/%d lambda values for depth=%d, e=%d%n",
                      neededLambdas.size(),
                      lambdas.length,
                      depth,
                      e);

    for (double lambda : neededLambdas) {
      String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, termFilter, depth, lambda, e);
      String trecRunPath = trecRunFolder + "/" + runName;

      new FileWriter(trecRunPath, false).close();
    }

    Map<Double, List<String>> resultsPerLambda = new ConcurrentHashMap<>();
    for (double lambda : neededLambdas) {
      resultsPerLambda.put(lambda, new CopyOnWriteArrayList<>());
    }

    boolean useParallel = true;
    final String parallelMode;

    int firstQueryId = topics.isEmpty() ? -1 : Integer.parseInt(topics.get(0).num);

    if (rerankMethod.equals("monot5") || (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB"))) {
      if ((monoT5Cache != null && !monoT5Cache.containsQuery(firstQueryId)) || trecRunFolder.contains("test")) {
        useParallel = false;
        parallelMode = "sequential (MonoT5 cache cold for first topic or test run)";
        System.out.println("⚠ MonoT5 cache cold for first topic or test run - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
      if ((vllmCache != null && !vllmCache.containsQuery(firstQueryId)) || trecRunFolder.contains("test")) {
        useParallel = false;
        parallelMode = "sequential (VLLM cache cold for first topic or test run)";
        System.out.println("⚠ VLLM cache cold for first topic or test run - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else if (rfStrategy.equals("VLLM-NARR") || rfStrategy.equals("VLLM-NARR-PROB")) {
      if ((vllmNarrCache != null && !vllmNarrCache.containsQuery(firstQueryId)) || trecRunFolder.contains("test")) {
        useParallel = false;
        parallelMode = "sequential (VLLM-NARR cache cold for first topic or test run)";
        System.out.println("⚠ VLLM-NARR cache cold for first topic or test run - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else if (rfStrategy.equals("VLLM-JUDGESPANS")) {
      if ((vllmCache != null && !vllmCache.containsQuery(firstQueryId)) || trecRunFolder.contains("test")) {
        useParallel = false;
        parallelMode = "sequential (VLLM judge+spans cache cold for first topic or test run)";
        System.out.println("⚠ VLLM judge+spans cache cold for first topic or test run - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else if (rfStrategy.equals("VLLM-NARR-JUDGESPANS")) {
      if ((vllmNarrCache != null && !vllmNarrCache.containsQuery(firstQueryId)) || trecRunFolder.contains("test")) {
        useParallel = false;
        parallelMode = "sequential (VLLM-NARR judge+spans cache cold for first topic or test run)";
        System.out.println("⚠ VLLM-NARR judge+spans cache cold for first topic or test run - using SEQUENTIAL processing to avoid overload");
      } else {
        parallelMode = "parallel";
      }
    } else {
      parallelMode = "parallel";
    }

    if (!termFilter.equals("none") && termsProvider != null
        && (!termsProvider.containsQuery(firstQueryId) || trecRunFolder.contains("test"))) {
      useParallel = false;
      System.out.println("⚠ Terms provider cache cold or test run - using SEQUENTIAL processing to avoid overload");
    }

    Stream<Topic> topicStream = useParallel ? topics.parallelStream() : topics.stream();

    topicStream.forEach(topic -> {
      try {
        StatsProvider threadStatsProvider = getThreadLocalStatsProvider();

        String queryStr = topic.title;

        QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);
        Query query = parser.parse(QueryParser.escape(queryStr));

        TopDocs results = searcher.search(query, 1000);

        if (rerankMethod.equals("none")) {
          for (double lambda : neededLambdas) {
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, termFilter, depth, lambda, e);

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
          for (double lambda : neededLambdas) {
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, termFilter, depth, lambda, e);

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
          TermWeights expandedQueryWeights = queryExpansion(queryStr,
                                                            topic.narrative,
                                                            Integer.parseInt(topic.num),
                                                            results,
                                                            rfStrategy,
                                                            termFilter,
                                                            searcher,
                                                            threadStatsProvider,
                                                            rfModel,
                                                            depth,
                                                            e,
                                                            monoT5Cache,
                                                            vllmCache,
                                                            vllmNarrCache,
                                                            termsProvider);

          List<String> processedTerms = new ArrayList<>();
          try (TokenStream tokenStream = analyzer.tokenStream(SEARCH_FIELD, queryStr)) {
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
              processedTerms.add(tokenStream.getAttribute(CharTermAttribute.class).toString());
            }
            tokenStream.end();
          }
          TermWeights originalQueryWeights = TermWeights.fromTerms(processedTerms).scaleToL1Norm();

          for (double lambda : neededLambdas) {
            String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, termFilter, depth, lambda, e);

            TermWeights finalQuery = TermWeights.interpolate(originalQueryWeights, expandedQueryWeights, lambda);

            StringBuilder queryBuilder = new StringBuilder();
            finalQuery.forEach((term, weight) -> {
              String escapedTerm = QueryParser.escape(term);
              queryBuilder.append(escapedTerm).append("^").append(String.format("%.10f ", weight));
            });

            Query expandedQuery = parser.parse(queryBuilder.toString());
            TopDocs expandedResults = searcher.search(expandedQuery, 1000);

            if (expandedResults != null) {
              StringBuilder resultStr = new StringBuilder();
              for (int i = 0; i < Math.min(1000, expandedResults.scoreDocs.length); i++) {
                ScoreDoc scoreDoc = expandedResults.scoreDocs[i];
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                String docno = doc.get(DOCID_FIELD);
                String tag = (i == 0) ? runName : "--";
                resultStr.append(String.format("%s Q0 %s %d %.6f %s\n", topic.num, docno, i + 1, scoreDoc.score, tag));
              }
              resultsPerLambda.get(lambda).add(resultStr.toString());
            }
          }
        }

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

    System.out.println("Writing results to files...");
    for (double lambda : neededLambdas) {
      String runName = buildRunName(baselineModel, rerankMethod, rfStrategy, rfModel, termFilter, depth, lambda, e);
      String trecRunPath = trecRunFolder + "/" + runName;

      try (BufferedWriter runWriter = new BufferedWriter(new FileWriter(trecRunPath, false))) {
        for (String result : resultsPerLambda.get(lambda)) {
          runWriter.write(result);
        }
      }
    }
    System.out.printf("✓ Completed depth=%d, e=%d with %d lambda values%n", depth, e, neededLambdas.size());

  }

  private static TopDocs rerankWithMonoT5(String queryText, String narrative, int queryId, TopDocs initialResults,
                                          IndexSearcher searcher, int depth, LLMCache cache) throws IOException {

    List<ScoredDoc> scoredDocs = new ArrayList<>();
    int docsToRerank = Math.min(depth, initialResults.scoreDocs.length);

    for (int i = 0; i < docsToRerank; i++) {
      ScoreDoc sd = initialResults.scoreDocs[i];
      Document doc = searcher.storedFields().document(sd.doc);
      String docText = doc.get(SEARCH_FIELD);

      LLMResult result = cache.get(queryId, sd.doc, queryText, narrative, docText);

      scoredDocs.add(new ScoredDoc(sd.doc, result.score));
    }

    for (int i = docsToRerank; i < initialResults.scoreDocs.length; i++) {
      ScoreDoc sd = initialResults.scoreDocs[i];
      scoredDocs.add(new ScoredDoc(sd.doc, -1000 - i));
    }

    scoredDocs.sort((a, b) -> Double.compare(b.score, a.score));

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
                                                              LLMCache vllmCache, LLMCache vllmNarrCache) throws IOException {
    switch (rfStrategy) {
      case "PRF":
        Map<Integer, Double> relevantDocs = new HashMap<>();
        for (int i = 0; i < Math.min(k, results.scoreDocs.length); i++) {
          ScoreDoc sd = results.scoreDocs[i];
          relevantDocs.put(sd.doc, (double) sd.score);
        }

        return relevantDocs;
      case "ORACLE-K":
        return filterWithOracle(queryid, results, k, k);
      case "MONOT5":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             monoT5Cache,
                             (sd, result) -> (double) sd.score);
      case "MONOT5-PROB":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             monoT5Cache,
                             (sd, result) -> result.probTrue);
      case "VLLM":
        return filterWithLLM(queryid,
                             queryText,
                             null,
                             results,
                             k,
                             searcher,
                             vllmCache,
                             (sd, result) -> (double) sd.score);
      case "VLLM-PROB":
        return filterWithLLM(queryid,
                             queryText,
                             null,
                             results,
                             k,
                             searcher,
                             vllmCache,
                             (sd, result) -> result.probTrue);
      case "VLLM-NARR":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             vllmNarrCache,
                             (sd, result) -> (double) sd.score);
      case "VLLM-NARR-PROB":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             vllmNarrCache,
                             (sd, result) -> result.probTrue);
      case "VLLM-JUDGESPANS":
        return filterWithLLM(queryid,
                             queryText,
                             null,
                             results,
                             k,
                             searcher,
                             vllmCache,
                             (sd, result) -> (double) sd.score);
      case "VLLM-NARR-JUDGESPANS":
        return filterWithLLM(queryid,
                             queryText,
                             narrative,
                             results,
                             k,
                             searcher,
                             vllmNarrCache,
                             (sd, result) -> (double) sd.score);
      default:
        throw new IllegalArgumentException("Unknown RF strategy: " + rfStrategy);
    }
  }

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

      LLMResult result = llmCache.get(queryid, sd.doc, queryText, narrative, docText);

      if (result.isRelevant) {
        filteredDocs.put(sd.doc, scoreFunction.getScore(sd, result));
      }
    }

    return filteredDocs;
  }

  private static Map<Integer, Double> filterWithOracle(int queryid, TopDocs results, int depth, int maxDocs) {
    Map<Integer, Double> oracleDocs = new HashMap<>();

    if (!oracle.containsKey(queryid)) {
      return oracleDocs;
    }

    Set<Integer> relevantDocIds = oracle.get(queryid);
    int count = 0;

    for (int i = 0; i < Math.min(depth, results.scoreDocs.length); i++) {
      ScoreDoc sd = results.scoreDocs[i];
      if (relevantDocIds.contains(sd.doc)) {
        oracleDocs.put(sd.doc, (double) sd.score);
        count++;
        if (count >= maxDocs) {
          break;
        }
      }
    }

    return oracleDocs;
  }

  private static TermWeights queryExpansion(String originalQuery, String narrative, int queryId, TopDocs results,
                                            String rfStrategy, String termFilter, IndexSearcher searcher,
                                            StatsProvider statsProvider, String rfModel, int k, int e,
                                            LLMCache monoT5Cache, LLMCache vllmCache, LLMCache vllmNarrCache,
                                            TermsProvider termsProvider) throws IOException {
    Map<Integer, Double> prfDocs = filterRelevantDocuments(queryId,
                                                           originalQuery,
                                                           narrative,
                                                           results,
                                                           rfStrategy,
                                                           k,
                                                           searcher,
                                                           monoT5Cache,
                                                           vllmCache,
                                                           vllmNarrCache);
    Smoothing smoothing = new AdditiveSmoothing(smoothingParameter, SEARCH_FIELD, statsProvider);

    boolean isSpanLM = termFilter.equals("vllmspans2") || termFilter.equals("vllmspans2-nonarr")
        || termFilter.equals("vllmjudgespans");
    Map<Integer, Set<String>> allowedTerms = null;
    Map<Integer, Map<String, Integer>> spanTermFreqs = null;
    if (!termFilter.equals("none") && !isSpanLM && termsProvider != null) {
      allowedTerms = new HashMap<>();
      for (Map.Entry<Integer, Double> entry : prfDocs.entrySet()) {
        int docId = entry.getKey();
        Document doc = searcher.storedFields().document(docId);
        String docText = doc.get(SEARCH_FIELD);
        List<String> rawTerms = termsProvider.get(queryId, docId, originalQuery, narrative, docText);
        Set<String> analyzed = analyzeAndIntersect(rawTerms, docId, smoothing);
        if (!analyzed.isEmpty()) {
          allowedTerms.put(docId, analyzed);
        }
      }
    } else if (isSpanLM && termsProvider != null) {
      spanTermFreqs = new HashMap<>();
      for (Map.Entry<Integer, Double> entry : prfDocs.entrySet()) {
        int docId = entry.getKey();
        Document doc = searcher.storedFields().document(docId);
        String docText = doc.get(SEARCH_FIELD);
        List<String> spans = termsProvider.get(queryId, docId, originalQuery, narrative, docText);
        Map<String, Integer> tf = analyzeSpansToTF(spans, smoothing);
        if (!tf.isEmpty()) {
          spanTermFreqs.put(docId, tf);
        }
      }
    }

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

    List<String> queryTerms = extractQueryTerms(originalQuery);

    TermWeights weights = (spanTermFreqs != null)
        ? feedbackModel.getTermWeightsFromSpans(prfDocs, queryTerms, spanTermFreqs)
        : feedbackModel.getTermWeights(prfDocs, queryTerms, allowedTerms);
    return weights.pruneToSize(e).scaleToL1Norm();
  }

  private static Set<String> analyzeAndIntersect(List<String> rawTerms, int docId, Smoothing smoothing) {
    Set<String> docVocab = smoothing.getDocTerms(docId);
    Set<String> analyzed = new HashSet<>();
    for (String raw : rawTerms) {
      try (TokenStream ts = analyzer.tokenStream(SEARCH_FIELD, raw)) {
        ts.reset();
        while (ts.incrementToken()) {
          String term = ts.getAttribute(CharTermAttribute.class).toString();
          if (docVocab.contains(term)) {
            analyzed.add(term);
          }
        }
        ts.end();
      } catch (IOException ex) {
      }
    }
    return analyzed;
  }

  private static Map<String, Integer> analyzeSpansToTF(List<String> spans, Smoothing smoothing) {
    Map<String, Integer> tf = new HashMap<>();
    for (String span : spans) {
      try (TokenStream ts = analyzer.tokenStream(SEARCH_FIELD, span)) {
        ts.reset();
        while (ts.incrementToken()) {
          String term = ts.getAttribute(CharTermAttribute.class).toString();
          if (smoothing.termExists(term)) {
            tf.merge(term, 1, Integer::sum);
          }
        }
        ts.end();
      } catch (IOException ex) {
      }
    }
    return tf;
  }

  private static Map<Integer, Set<Integer>> loadOracleRelevance(String pathToQrelsFile,
                                                                IndexSearcher searcher) throws IOException, ParseException {
    Map<Integer, Set<Integer>> oracleRelevance = new HashMap<>();
    try (BufferedReader br = new BufferedReader(new FileReader(pathToQrelsFile))) {
      String line;
      while ((line = br.readLine()) != null) {
        String[] parts = line.trim().split("\s+");
        if (parts.length >= 4) {
          Integer queryId = Integer.parseInt(parts[0]);
          String docno = parts[2];
          int relevance = Integer.parseInt(parts[3]);
          if (relevance > 0) {
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

  private static class MethodSpec {
    final String label;
    final String strategy;
    final String termFilter;
    final String rfModel;
    final int depth;
    final int e;
    final double lambda;

    MethodSpec(String label, String strategy, String termFilter, String rfModel,
              int depth, int e, double lambda) {
      this.label = label;
      this.strategy = strategy;
      this.termFilter = termFilter;
      this.rfModel = rfModel;
      this.depth = depth;
      this.e = e;
      this.lambda = lambda;
    }
  }

  @FunctionalInterface
  private interface IOSupplier<T> {
    T get() throws IOException;
  }

  private static <T> T unchecked(IOSupplier<T> supplier) {
    try {
      return supplier.get();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static List<MethodSpec> parseCompareSpecs(String compareSpec) {
    List<MethodSpec> specs = new ArrayList<>();
    for (String entry : compareSpec.split(";")) {
      if (entry.trim().isEmpty()) continue;
      String[] parts = entry.split(":", 7);
      if (parts.length != 7) {
        throw new IllegalArgumentException(
            "Invalid --compare entry (expected label:strategy:term_filter:rf_model:depth:e:lambda): " + entry);
      }
      specs.add(new MethodSpec(parts[0], parts[1], parts[2], parts[3],
          Integer.parseInt(parts[4]), Integer.parseInt(parts[5]), Double.parseDouble(parts[6])));
    }
    return specs;
  }

  private static List<Map.Entry<String, Double>> sortedTopN(TermWeights weights, int n) {
    List<Map.Entry<String, Double>> entries = new ArrayList<>();
    weights.forEach((term, w) -> entries.add(new AbstractMap.SimpleEntry<>(term, w)));
    entries.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
    return entries.subList(0, Math.min(n, entries.size()));
  }

  private static String escapeLatex(String s) {
    return s.replace("\\", "\\textbackslash ")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("_", "\\_");
  }

  private static void runDumpTerms(List<Topic> topics, IndexSearcher searcher, StatsProvider statsProvider,
                                   String topicId, String compareSpec, int topN, String cacheDir,
                                   String qrelsPath) throws Exception {
    Topic topic = topics.stream().filter(t -> t.num.equals(topicId)).findFirst()
        .orElseThrow(() -> new IllegalArgumentException("Topic not found: " + topicId));

    List<MethodSpec> specs = parseCompareSpecs(compareSpec);

    if (qrelsPath != null && oracle == null && specs.stream().anyMatch(s -> s.strategy.contains("ORACLE"))) {
      System.out.println("Parsing qrels for oracle spec: " + qrelsPath);
      oracle = loadOracleRelevance(qrelsPath, searcher);
    }

    String queryStr = topic.title;
    QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);
    Query query = parser.parse(QueryParser.escape(queryStr));
    TopDocs results = searcher.search(query, 1000);
    TermWeights originalQueryWeights = TermWeights.fromTerms(extractQueryTerms(queryStr)).scaleToL1Norm();

    Map<String, LLMCache> llmCacheByStrategy = new HashMap<>();
    Map<String, VLLMJudgeSpansCache> judgeSpansCacheByStrategy = new HashMap<>();
    Map<String, TermsProvider> termsProviderByFilter = new HashMap<>();

    StringBuilder out = new StringBuilder();
    out.append("\\begin{table*}[t]\n\\centering\n");
    out.append("\\caption{}\n");
    out.append(String.format("\\label{tab:terms-%s}\n", topic.num));

    double subWidth = 0.95 / Math.min(specs.size(), 3);
    for (int si = 0; si < specs.size(); si++) {
      MethodSpec spec = specs.get(si);

      LLMCache monoT5 = null;
      LLMCache vllm = null;
      LLMCache vllmNarr = null;
      TermsProvider tp = null;

      switch (spec.strategy) {
        case "MONOT5": case "MONOT5-PROB":
          monoT5 = llmCacheByStrategy.computeIfAbsent("MONOT5",
              k -> unchecked(() -> new MonoT5Cache(cacheDir)));
          break;
        case "VLLM": case "VLLM-PROB":
          vllm = llmCacheByStrategy.computeIfAbsent("VLLM",
              k -> unchecked(() -> new VLLMCache(cacheDir)));
          break;
        case "VLLM-NARR": case "VLLM-NARR-PROB":
          vllmNarr = llmCacheByStrategy.computeIfAbsent("VLLM-NARR",
              k -> unchecked(() -> new VLLMCache(cacheDir, "vllm_narr_cache.tsv")));
          break;
        case "VLLM-JUDGESPANS": {
          VLLMJudgeSpansCache jsc = judgeSpansCacheByStrategy.computeIfAbsent("VLLM-JUDGESPANS",
              k -> unchecked(() -> new VLLMJudgeSpansCache(cacheDir, false)));
          vllm = jsc.asLLMCache();
          if (spec.termFilter.equals("vllmjudgespans")) tp = jsc.asTermsProvider();
          break;
        }
        case "VLLM-NARR-JUDGESPANS": {
          VLLMJudgeSpansCache jsc = judgeSpansCacheByStrategy.computeIfAbsent("VLLM-NARR-JUDGESPANS",
              k -> unchecked(() -> new VLLMJudgeSpansCache(cacheDir, true)));
          vllmNarr = jsc.asLLMCache();
          if (spec.termFilter.equals("vllmjudgespans")) tp = jsc.asTermsProvider();
          break;
        }
        default:
      }

      if (tp == null && !spec.termFilter.equals("none") && !spec.termFilter.equals("vllmjudgespans")) {
        final String tf = spec.termFilter;
        tp = termsProviderByFilter.computeIfAbsent(tf, k -> {
          switch (k) {
            case "vllmspans2":
              return unchecked(() -> new VLLMSpansCache(cacheDir));
            case "vllmspans2-nonarr":
              return unchecked(() -> new VLLMSpansCache(cacheDir, false));
            default:
              throw new IllegalArgumentException("Unknown term_filter: " + k);
          }
        });
      }

      TermWeights expanded = queryExpansion(queryStr, topic.narrative, Integer.parseInt(topic.num), results,
          spec.strategy, spec.termFilter, searcher, statsProvider, spec.rfModel, spec.depth, spec.e,
          monoT5, vllm, vllmNarr, tp);

      TermWeights finalWeights = TermWeights.interpolate(originalQueryWeights, expanded, spec.lambda);
      List<Map.Entry<String, Double>> topTerms = sortedTopN(finalWeights, topN);

      out.append(String.format("\\begin{subtable}[t]{%.3f\\textwidth}\n\\centering\n\\small\n", subWidth));
      out.append(String.format("\\caption{%s}\n", escapeLatex(spec.label)));
      out.append("\\begin{tabular}{@{}lr@{}}\n\\toprule\nTerm & Weight \\\\\n\\midrule\n");
      for (Map.Entry<String, Double> entry : topTerms) {
        out.append(String.format("%s & %.4f \\\\\n", escapeLatex(entry.getKey()), entry.getValue()));
      }
      out.append("\\bottomrule\n\\end{tabular}\n\\end{subtable}");

      boolean lastInRow = (si + 1) % 3 == 0 || si == specs.size() - 1;
      if (lastInRow) {
        out.append("\n");
        if (si != specs.size() - 1) {
          out.append("\\par\\bigskip\n");
        }
      } else {
        out.append("%\n\\hfill\n");
      }
    }

    out.append("\\end{table*}\n");

    System.out.println();
    System.out.println(out);

    for (LLMCache c : llmCacheByStrategy.values()) c.close();
    for (VLLMJudgeSpansCache c : judgeSpansCacheByStrategy.values()) c.close();
    for (TermsProvider c : termsProviderByFilter.values()) c.close();
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
    System.err.println("  --rerank_method <method>      Reranking method: none|monot5|prf|dump_terms (default: none)");
    System.err.println("  --prf_strategy <strategy>     PRF strategy: PRF|ORACLE-K|MONOT5|MONOT5-PROB|VLLM|VLLM-PROB|");
    System.err.println("                                VLLM-NARR|VLLM-NARR-PROB|VLLM-JUDGESPANS|VLLM-NARR-JUDGESPANS");
    System.err.println("  --prf_model <model>           RF model: RM3|DMM|MEDMM (default: RM3)");
    System.err.println("  --term_filter <filter>        Term filter: none|vllmspans2|vllmspans2-nonarr|vllmjudgespans");
    System.err.println("  --rerank_depth <depth>        Reranking depth (default: 100)");
    System.err.println("  --lambda <lambda>             Lambda for interpolation (default: 0.0)");
    System.err.println("  -e <e>                        Number of expansion terms (default: 20)");

    System.err.println("\nGrid search arguments:");
    System.err.println("  --grid_search                 Enable grid search mode");
    System.err.println("  --depths <depths>             Comma-separated depth values");
    System.err.println("  --e_values <e_values>         Comma-separated e values");
    System.err.println("  --lambdas <lambdas>           Comma-separated lambda values");

    System.err.println("\ndump_terms mode arguments:");
    System.err.println("  --topic_id <id>                Topic number to dump expansion terms for");
    System.err.println("  --top_n <n>                    Number of top terms to show per method (default: 10)");
    System.err.println("  --compare <spec>                ';'-separated label:strategy:term_filter:rf_model:depth:e:lambda entries");

    System.err.println("\nConditional arguments (required for some modes):");
    System.err.println("  --qrels <path>                Path to qrels file (required for ORACLE-K strategy)");
    System.err.println("  --cache_dir <path>            Path to cache directory (required for LLM strategies)");
  }
}
