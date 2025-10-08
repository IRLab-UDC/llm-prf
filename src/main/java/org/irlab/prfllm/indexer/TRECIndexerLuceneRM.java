package org.irlab.prfllm.indexer;

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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;

public class TRECIndexerLuceneRM {

  public static void main(String[] args) {
    String dataset = "msmarco";
    String datasetPath = "/home/javier/data/datasets/" + dataset;
    String indexPath = "/home/javier/data/indices/" + dataset + "_index";


    for (int i = 0; i < args.length; i++) {
      if ("--dataset_path".equals(args[i])) {
        datasetPath = args[++i];
      } else if ("--index_path".equals(args[i])) {
        indexPath = args[++i];
      }
    }


    try {
      Directory dir = FSDirectory.open(Paths.get(indexPath));
      Analyzer analyzer = new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);

      // Configurar la similitud para usar un modelo de lenguaje (LM)
      Similarity similarity = new LMDirichletSimilarity();

      IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
      iwc.setSimilarity(similarity);
      iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

      IndexWriter writer = new IndexWriter(dir, iwc);

      System.out.println("Indexando documentos de: " + datasetPath);
      indexDocs(writer, new File(datasetPath));

      writer.close();
      System.out.println("Indexación completada.");

    } catch (IOException e) {
      System.err.println("Error al indexar: " + e.getMessage());
    }
  }

  private static void indexDocs(final IndexWriter writer, File file) throws IOException {
    if (file.isDirectory()) {
      for (File f : file.listFiles()) {
        indexDocs(writer, f);
      }
    } else {
      indexDocMsMarco(writer, file);
    }
  }

  private static class JsonDocument {
    String id;
    String contents;
  }

  private static void indexDocMsMarco(IndexWriter writer, File file) throws IOException {
    Gson gson = new Gson(); // Instancia de Gson para parsear JSON
    try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
      String jsonLine;
      int lineNumber = 0;

      FieldType customType = new FieldType(TextField.TYPE_STORED);
      customType.setStoreTermVectors(true);
      customType.setStoreTermVectorPositions(true);
      customType.setStoreTermVectorOffsets(true);

      // Leer línea por línea
      while ((jsonLine = reader.readLine()) != null) {
        lineNumber++;
        if (jsonLine.trim().isEmpty()) {
          continue; // Saltar líneas vacías
        }

        try {
          // 1. Parsear la línea JSON al objeto auxiliar
          JsonDocument jsonDoc = gson.fromJson(jsonLine, JsonDocument.class);

          // 2. Crear el documento de Lucene
          Document doc = new Document();

          // Campo "id": Lo almacenamos y no lo tokenizamos (StringField)
          doc.add(new StringField("DOCNO", jsonDoc.id, Field.Store.YES));

          // Campo "contents": Lo indexamos y tokenizamos para búsqueda (TextField)
          // También lo almacenamos para mostrar los resultados, si es necesario.
          doc.add(new Field("TEXT", jsonDoc.contents, customType));

          // 3. Añadir el documento al IndexWriter
          writer.addDocument(doc);
          System.out.println("Añadiendo DOC con ID: " + jsonDoc.id);

        } catch (Exception e) {
          // Capturar excepciones de parseo (JSON malformado) o Lucene.
          System.err.println("Error al procesar la línea "
                             + lineNumber
                             + " en el archivo "
                             + file.getName()
                             + ": "
                             + e.getMessage());
          // Puedes optar por lanzar la excepción o simplemente continuar con el siguiente documento.
        }
      }
    }
  }

  private static void indexDoc(IndexWriter writer, File file) throws IOException {
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
          docBuffer.append(line).append("\n");
          inDoc = false;
          // Parse and index this document
          Document doc = parseTrecDoc(docBuffer.toString(), file);
          if (doc != null) {
            writer.addDocument(doc);
          }
        } else if (inDoc) {
          docBuffer.append(line).append("\n");
        }
      }
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

    // Extract DOCNO
    String docno = extractTagContent(docString, "DOCNO");

    doc.add(new TextField("DOCNO", docno != null ? docno.trim() : "", Field.Store.YES));

    // Optionally extract TITLE, TEXT, etc.
    String title = extractTagContent(docString, "HEAD");
    if (title != null) {
      doc.add(new Field("HEAD", title.trim(), customType));
    }

    String text = extractTagContent(docString, "TEXT");
    if (text != null) {
      doc.add(new Field("CONTENT", text.trim(), customType));
    }

    // We create a new text field with the title+contents for better retrieval
    StringBuilder fullText = new StringBuilder();
    if (title != null) {
      fullText.append(title.trim()).append(" ");
    }
    if (text != null) {
      fullText.append(text.trim());
    }
    doc.add(new Field("TEXT", fullText.toString().trim(), customType));


    // Add filename and filepath for reference
    doc.add(new TextField("filename", file.getName(), Field.Store.YES));
    doc.add(new TextField("filepath", file.getAbsolutePath(), Field.Store.YES));

    return doc;
  }

  // Simple tag extractor for TREC tags
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
}