package org.irlab.ecir26.indexer;

import com.google.gson.Gson;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.jsoup.Jsoup;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicInteger;

public class TRECIndexerLuceneRM {

  // Global counter for total documents indexed.
  private static final AtomicInteger totalDocsIndexed = new AtomicInteger(0);

  public static void main(String[] args) {
    String datasetPath = null;
    String indexPath = null;

    for (int i = 0; i < args.length; i++) {
      if ("--dataset".equals(args[i])) {
        datasetPath = args[++i];
      } else if ("--index".equals(args[i])) {
        indexPath = args[++i];
      }
    }

    // Validate required arguments
    if (datasetPath == null || indexPath == null) {
      System.err.println("Error: Both --dataset and --index arguments are required.");
      System.err.println("Usage: java -jar <jar> --dataset <path> --index <path>");
      System.exit(1);
    }

    try {
      Directory dir = FSDirectory.open(Paths.get(indexPath));
      Analyzer analyzer = new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);

      // Configure similarity to use a language model (LM).
      Similarity similarity = new LMDirichletSimilarity();

      IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
      iwc.setSimilarity(similarity);
      iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

      IndexWriter writer = new IndexWriter(dir, iwc);

      System.out.println("=".repeat(80));
      System.out.println("Indexing documents from: " + datasetPath);
      System.out.println("Output index: " + indexPath);
      System.out.println("=".repeat(80));

      long startTime = System.currentTimeMillis();
      indexDocs(writer, new File(datasetPath));

      writer.close();

      long endTime = System.currentTimeMillis();
      long totalTime = (endTime - startTime) / 1000;

      System.out.println("=".repeat(80));
      System.out.println("✓ Indexing completed.");
      System.out.println("✓ Total documents indexed: " + totalDocsIndexed.get());
      System.out.println("✓ Total time: " + totalTime + " seconds");
      System.out.println("=".repeat(80));

    } catch (IOException e) {
      System.err.println("Error al indexar: " + e.getMessage());
    }
  }

  private static void indexDocs(final IndexWriter writer, File file) throws IOException {
    if (file.isDirectory()) {
      // If it is the info directory, ignore it.
      if (file.getName().equals("info")) {
        System.out.println("⊗ Skipping info directory: " + file.getAbsolutePath());
        return;
      }

      // Sort files alphabetically to ensure deterministic indexing order.
      File[] files = file.listFiles();
      if (files != null) {
        Arrays.sort(files, Comparator.comparing(File::getName));
        for (File f : files) {
          indexDocs(writer, f);
        }
      }
    } else {
      // Detect format: MS MARCO vs TREC.
      if (isMsMarcoFormat(file)) {
        indexDocMsMarco(writer, file);
      } else {
        indexDoc(writer, file);
      }
    }
  }

  // Helper method to detect if file is MS MARCO JSON format.
  private static boolean isMsMarcoFormat(File file) throws IOException {
    try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
      String firstLine = reader.readLine();
      if (firstLine == null || firstLine.trim().isEmpty()) {
        return false;
      }
      // MS MARCO files start with JSON objects containing "id" and "contents"
      // TREC files typically start with <DOC> or other XML tags
      return firstLine.trim().startsWith("{") && (firstLine.contains("\"id\"") || firstLine.contains("\"contents\""));
    }
  }

  private static class JsonDocument {
    String id;
    String contents;
  }

  private static void indexDocMsMarco(IndexWriter writer, File file) throws IOException {
    System.out.println("→ Processing MS MARCO file: " + file.getName());

    Gson gson = new Gson();
    int docsInFile = 0;

    try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
      String jsonLine;
      int lineNumber = 0;

      FieldType customType = new FieldType(TextField.TYPE_STORED);
      customType.setStoreTermVectors(true);
      customType.setStoreTermVectorPositions(true);
      customType.setStoreTermVectorOffsets(true);

      // Read line by line
      while ((jsonLine = reader.readLine()) != null) {
        lineNumber++;
        if (jsonLine.trim().isEmpty()) {
          continue; // Skip empty lines
        }

        try {
          JsonDocument jsonDoc = gson.fromJson(jsonLine, JsonDocument.class);

          Document doc = new Document();

          // Docid: store it and don't tokenize it
          doc.add(new StringField("docid", jsonDoc.id, Field.Store.YES));

          // Field "content": Index and tokenize for search (TextField)
          // Also store it to display results, if necessary.
          doc.add(new Field("content", jsonDoc.contents, customType));

          writer.addDocument(doc);
          docsInFile++;
          totalDocsIndexed.incrementAndGet();

        } catch (Exception e) {
          System.err.println("  ✗ Error processing line "
                             + lineNumber
                             + " in file "
                             + file.getName()
                             + ": "
                             + e.getMessage());
          throw new RuntimeException(e);
        }
      }

      System.out.println("  ✓ Indexed " + docsInFile + " documents from " + file.getName());
    }
  }

  private static void indexDoc(IndexWriter writer, File file) throws IOException {
    System.out.println("→ Processing TREC file: " + file.getAbsolutePath());

    int docsInFile = 0;

    try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
      StringBuilder docBuffer = new StringBuilder();
      String line;
      boolean inDoc = false;

      while ((line = reader.readLine()) != null) {
        if (line.trim().equalsIgnoreCase("<DOC>")) {
          inDoc = true;
          docBuffer.setLength(0);
          docBuffer.append(line).append("\n");
        } else if (line.trim().equalsIgnoreCase("</DOC>")) {
          // Only process if we were actually inside a document
          if (inDoc) {
            docBuffer.append(line).append("\n");
            inDoc = false;
            // Parse and index this document
            Document doc = parseTrecDoc(docBuffer.toString(), file);
            if (doc != null) {
              writer.addDocument(doc);
              docsInFile++;
              totalDocsIndexed.incrementAndGet();
            }
            // Clear buffer to prevent malformed data (e.g., duplicate </DOC> tags) from being re-indexed
            docBuffer.setLength(0);
          }
          // If inDoc is false, this is a stray </DOC> tag - ignore it
        } else if (inDoc) {
          docBuffer.append(line).append("\n");
        }
      }

      System.out.println("  ✓ Indexed " + docsInFile + " documents from " + file.getName());
    }
  }

  // Helper to parse a TREC <DOC> block into a Lucene Document
  private static Document parseTrecDoc(String docString, File file) {
    Document doc = new Document();
    // We create a custom field for textfield stored and with termvectors
    FieldType customType = new FieldType(TextField.TYPE_STORED);
    customType.setStoreTermVectors(true);
    customType.setStoreTermVectorPositions(true);
    customType.setStoreTermVectorOffsets(true);

    // Extract DOCNO (raw, no cleaning)
    String docno = extractTagContent(docString, "DOCNO");

    doc.add(new TextField("docid", docno != null ? docno.trim() : "", Field.Store.YES));

    // Detect if this is a web collection (WT10G) based on DOCNO format
    boolean isWebCollection = docno != null && docno.trim().startsWith("WTX");
    StringBuilder fullText = new StringBuilder();

    // Extract TITLE/HEAD (raw)
    String title = extractTagContent(docString, "HEAD");
    if (title != null) {
      // Clean HTML from title if web collection
      String cleanTitle = isWebCollection ? cleanHtmlContent(title) : cleanRobustContent(title);
      fullText.append(cleanTitle.trim()).append(" ");
    }

    // Extract TEXT/CONTENT (raw)
    String text = extractTagContent(docString, "TEXT");
    if (text != null) {
      // Clean HTML from text if web collection
      String cleanText = isWebCollection ? cleanHtmlContent(text) : cleanRobustContent(text);
      fullText.append(cleanText.trim());
    } else {
      // In the case of web collections there is no explicit TEXT tag, the content is everything from the end of the header </DOCHDR> until the end of the doc </DOC>
      String headerEndTag = "</DOCHDR>";
      int headerEndIndex = docString.indexOf(headerEndTag);
      int docEndIndex = docString.indexOf("</DOC>");
      if (headerEndIndex != -1 && docEndIndex != -1 && docEndIndex > headerEndIndex) {
        text = docString.substring(headerEndIndex + headerEndTag.length(), docEndIndex);
        String cleanText = isWebCollection ? cleanHtmlContent(text) : cleanRobustContent(text);
        doc.add(new Field("CONTENT", cleanText.trim(), customType));
        fullText.append(cleanText.trim());
      }
    }

    doc.add(new Field("content", fullText.toString().trim(), customType));

    return doc;
  }

  // Simple tag extractor for TREC tags (NO cleaning - returns raw content)
  private static String extractTagContent(String doc, String tag) {
    String startTag = "<" + tag + ">";
    String endTag = "</" + tag + ">";
    int start = doc.indexOf(startTag);
    int end = doc.indexOf(endTag);
    if (start != -1 && end != -1 && end > start) {
      return doc.substring(start + startTag.length(), end);
    }
    return null;
  }

  // Clean HTML content using Jsoup (for web collections like WT10G)
  private static String cleanHtmlContent(String content) {
    if (content == null) {
      return null;
    }
    // Use Jsoup to parse HTML and extract clean text
    // Removes all tags, scripts, styles, and decodes entities
    return Jsoup.parse(content).text();
  }

  // Clean ROBUST04 content (remove HTML comments but keep structure)
  private static String cleanRobustContent(String content) {
    if (content == null) {
      return null;
    }
    // Remove HTML comments like <!-- PJG STAG 4700 -->
    // Remove basic HTML tags but preserve text structure
    return content.replaceAll("<!--.*?-->", "").replaceAll("<[^>]+>", "");
  }
}