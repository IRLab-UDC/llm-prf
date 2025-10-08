package org.irlab.prfllm.searcher;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
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

    private static Map<Integer, Set<Integer>> loadOracleRelevance(String pathToQrelsFile, IndexSearcher searcher)
            throws FileNotFoundException, IOException, ParseException {
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

    public static void main(String[] args) throws Exception {
        String index = "ap8889_index";  
        String topic_set = "topics.101-200";
        String qrelsFile = "qrels_ap8889_101_200.txt";
        // index = "robust4_index";
        // topic_set = "topics.301-350.trec.txt";
        // qrelsFile = "qrels.robust04.300-450.601-700.trec.txt";
        String indexPath = "/home/javier/data/indices/"+index;
        String topicsPath = "/home/javier/data/topics/"+topic_set;
        String qrelsPath = "/home/javier/data/topics/"+qrelsFile;
        String trecRunFolder = "/home/javier/data/runs/"+index;
        String cacheDir = "/home/javier/data/cache/"+index; // Collection-specific cache directory

        String searchBy = "title_only";
        float dirichletMu = 2000f;
        int rerankDepth = 100; // How many docs to rerank with MonoT5
        int e = 20;
        String prfSmoothingModel = "Additive"; // or "Dirichlet"
        Double prfSmoothingParameter = 0.1; // gamma for Additive, mu for Dirichlet
        String rfStrategy = "MONOT5"; // or "PRF", "ORACLE", "OLLAMA"
        String rfModel = "RM3";
        String rerankMethod = "prf"; // "none", "prf", or "monot5"
        String ollamaModel = "llama3.1:8b-instruct-fp16"; // Model to use for Ollama
        Double lambda = 0.7;
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
                case "--mu":
                    dirichletMu = Float.parseFloat(args[++i]);
                    break;
                case "--rf_model":
                    rfModel = args[++i];
                    break;
                case "--rf_strategy":
                    rfStrategy = args[++i];
                    break;
                case "--rerank_method":
                    rerankMethod = args[++i];
                    break;
                case "--rerank_depth":
                    rerankDepth = Integer.parseInt(args[++i]);
                    break;
                case "--serch_by":
                    searchBy = args[++i];
                    break;
                case "--prf_smoothing_model":
                    prfSmoothingModel = args[++i];
                    break;
                case "--prf_smoothing_parameter":
                    prfSmoothingParameter = Double.parseDouble(args[++i]);
                    break;
                case "--lambda":
                    lambda = Double.parseDouble(args[++i]);
                    break;
                case "-e":
                    e = Integer.parseInt(args[++i]);
                    break;
            }
        }
        // we build the runpath name from the parameters:
        String runName = "";
        if (rerankMethod.equals("monot5")) {
            runName = String.format("LMDirichlet-%.0f_%s_rerank-monot5_depth-%d",
                    dirichletMu, searchBy, rerankDepth);
        } else if (rerankMethod.equals("prf")) {
        
                runName = String.format(
                        "LMDirichlet-%.0f_%s_prf-%s_rfStrategy-%s_rfModel-%s_prfSmoothing-%s-%.4f_topK-%d_lambda-%.2f_e-%d",
                        dirichletMu, searchBy, true, rfStrategy, rfModel, prfSmoothingModel, prfSmoothingParameter,
                        rerankDepth,
                        lambda, e);
            
        } else {
            runName = String.format("LMDirichlet-%.0f_%s",
                    dirichletMu, searchBy);
        }

        String trecRunPath = trecRunFolder + "/" + runName;

        // Check if output file already exists
        java.io.File outputFile = new java.io.File(trecRunPath);
        if (outputFile.exists()) {
            System.out.println("Output file already exists: " + trecRunPath);
            System.out.println("Skipping this configuration.");
            return;
        }

        // Open index
        IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
        IndexSearcher searcher = new IndexSearcher(reader);
        oracle = loadOracleRelevance(qrelsPath, searcher);
        searcher.setSimilarity(new LMDirichletSimilarity(dirichletMu));

        // Configure Ollama model if using OLLAMA strategy
        if (rfStrategy.equals("OLLAMA")) {
            OllamaScorer.setModel(ollamaModel);
        }

        // Parse topics
        List<Topic> topics = parseTRECTopics(topicsPath);

        // Prepare output
        BufferedWriter runWriter = new BufferedWriter(new FileWriter(trecRunPath));
        StatsProvider statsProvider = new StatsProvider(searcher.getIndexReader());

        // Initialize caches once for all topics based on RF strategy
        MonoT5Cache monoT5Cache = null;
        OllamaCache ollamaCache = null;
        VLLMCache vllmCache = null;

        if (rerankMethod.equals("prf")) {
            if (rfStrategy.equals("MONOT5") || rfStrategy.equals("MONOT5-PROB")) {
                monoT5Cache = new MonoT5Cache(cacheDir);
            } else if (rfStrategy.equals("OLLAMA")) {
                ollamaCache = new OllamaCache(cacheDir, ollamaModel);
            } else if (rfStrategy.equals("VLLM") || rfStrategy.equals("VLLM-PROB")) {
                vllmCache = new VLLMCache(cacheDir);
            }
        } else if (rerankMethod.equals("monot5")) {
            monoT5Cache = new MonoT5Cache(cacheDir);
        }

        for (Topic topic : topics) {
            String queryStr = searchBy.equals("title_plus_description")
                    ? topic.title + " " + topic.description
                    : topic.title;

            QueryParser parser = new QueryParser(SEARCH_FIELD, analyzer);
            Query query = parser.parse(QueryParser.escape(queryStr));

            TopDocs results = searcher.search(query, 1000);
            System.out.println("Topic " + topic.num + " (" + topic.title + "): " + results.totalHits + " hits");

            if (rerankMethod.equals("monot5")) {
                // Direct reranking with MonoT5 (no query expansion)
                TopDocs rerankedResults = rerankWithMonoT5(queryStr, Integer.parseInt(topic.num), results, searcher,
                        rerankDepth, monoT5Cache);
                writeTrecRun(runWriter, topic.num, rerankedResults, searcher, runName);
            } else if (rerankMethod.equals("prf")) {
                // PRF with query expansion
                TermWeights expandedQueryWeights = queryExpansion(queryStr, Integer.parseInt(topic.num), results,
                        rfModel, rfStrategy, searcher,
                        statsProvider, prfSmoothingModel, prfSmoothingParameter.doubleValue(), lambda, rerankDepth, e,
                        ollamaModel, monoT5Cache, ollamaCache, vllmCache);

                // Second round with expanded query
                StringBuilder expandedQueryBuilder = new StringBuilder();
                expandedQueryWeights.forEach((term, weight) -> {
                    expandedQueryBuilder.append(term).append("^").append(String.format("%.4f ", weight));

                });
                System.out.println("Expanded Query: " + expandedQueryBuilder.toString().trim());
                Query expandedQuery = parser.parse(QueryParser.escape(expandedQueryBuilder.toString()));
                TopDocs expandedResults = searcher.search(expandedQuery, 1000);

                writeTrecRun(runWriter, topic.num, expandedResults, searcher, runName);

            } else {
                // No PRF, just write original results
                writeTrecRun(runWriter, topic.num, results, searcher, runName);
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

        runWriter.close();
        reader.close();
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
            writer.write(String.format("%s Q0 %s %d %f %s\n", qid, docno, rank, sd.score, runTag));
            rank++;
        }
    }

    // Rerank top results using MonoT5
    private static TopDocs rerankWithMonoT5(String queryText, int queryId, TopDocs initialResults,
            IndexSearcher searcher, int depth, MonoT5Cache cache) throws IOException {

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
            String rfStrategy, int k, IndexSearcher searcher, String ollamaModel,
            MonoT5Cache monoT5Cache, OllamaCache ollamaCache, VLLMCache vllmCache)
            throws IOException {
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
                    if (oracle.get(queryid).contains(sd.doc)) {
                        oracleDocs.put(sd.doc, (double) sd.score);

                    }
                }
                return oracleDocs;
            case "ORACLE-K":
                Map<Integer, Double> oracleKDocs = new HashMap<>();
                int count = 0;
                for (ScoreDoc sd : results.scoreDocs) {
                    if (oracle.get(queryid).contains(sd.doc)) {
                        oracleKDocs.put(sd.doc, (double) sd.score);
                        count++;
                        if (count >= k) {
                            break; // Stop after collecting k relevant documents
                        }
                    }
                }
                return oracleKDocs;
            case "MONOT5":
                // Use MonoT5 model as a binary filter: iterate through documents in rank order
                // and collect the first k documents that MonoT5 classifies as relevant.
                Map<Integer, Double> monoT5Docs = new HashMap<>();

                for (int i = 0; i < Math.min(k,results.scoreDocs.length); i++) {
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

                for (int i = 0; i < Math.min(k,results.scoreDocs.length); i++) {
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

                for (int i = 0; i < Math.min(k,results.scoreDocs.length); i++) {
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

    private static RelevanceFeedback getRelevanceFeedbackModel(String modelName, String field,
            Smoothing smoothing) {
        switch (modelName) {
            case "RM3":
                return new RM3(field, smoothing);
            default:
                throw new IllegalArgumentException("Unknown RF model: " + modelName);
        }
    }

    private static Smoothing geSmoothing(String modelName, double parameter, String field,
            StatsProvider statsProvider) {
        switch (modelName) {
            case "Dirichlet":
                return new DirichletSmoothing(parameter, field, statsProvider);
            case "Additive":
                return new AdditiveSmoothing(parameter, field, statsProvider);
            default:
                throw new IllegalArgumentException("Unknown smoothing model: " + modelName);
        }
    }

    private static TermWeights queryExpansion(String originalQuery, int queryId, TopDocs results, String rfModel,
            String rfStrategy,
            IndexSearcher searcher, StatsProvider statsProvider,
            String prfSmoothingModel, double prfSmoothingParameter, double lambda, int k, int e, String ollamaModel,
            MonoT5Cache monoT5Cache, OllamaCache ollamaCache, VLLMCache vllmCache)
            throws IOException {

        Map<Integer, Double> prfDocs = filterRelevantDocuments(queryId, originalQuery, results, rfStrategy, k,
                searcher, ollamaModel, monoT5Cache, ollamaCache, vllmCache);
        Smoothing smoothing = geSmoothing(prfSmoothingModel, prfSmoothingParameter, SEARCH_FIELD, statsProvider);
        RelevanceFeedback feedbackModel = getRelevanceFeedbackModel(rfModel, prfSmoothingModel, smoothing);
        TermWeights termWeights = feedbackModel.getTermWeights(prfDocs).pruneToSize(e).scaleToL1Norm();
        List<String> processedTerms = new ArrayList<>();
        try (TokenStream tokenStream = analyzer.tokenStream(SEARCH_FIELD, originalQuery)) {
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
                processedTerms.add(tokenStream.getAttribute(CharTermAttribute.class).toString());
            }
            tokenStream.end();
        }
        TermWeights originalQueryWeights = TermWeights.fromTerms(processedTerms).scaleToL1Norm();
        TermWeights expandedQuery = TermWeights.interpolate(originalQueryWeights, termWeights, lambda);
        return expandedQuery;
    }

}
