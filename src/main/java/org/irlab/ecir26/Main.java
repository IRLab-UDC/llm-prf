package org.irlab.ecir26;

import org.irlab.ecir26.indexer.TRECIndexerLuceneRM;
import org.irlab.ecir26.searcher.TRECSearcherLucene;

public class Main {

  public static void main(String[] args) {
    if (args.length == 0) {
      printUsage();
      System.exit(1);
    }

    String command = args[0].toLowerCase();

    // Create new args array without the first command argument
    String[] commandArgs = new String[args.length - 1];
    System.arraycopy(args, 1, commandArgs, 0, args.length - 1);

    switch (command) {
      case "index":
        System.out.println("Starting TREC Indexer...");
        try {
          TRECIndexerLuceneRM.main(commandArgs);
        } catch (Exception e) {
          System.err.println("Error during indexing: " + e.getMessage());
          e.printStackTrace();
          System.exit(1);
        }
        break;

      case "search":
        System.out.println("Starting TREC Searcher...");
        try {
          TRECSearcherLucene.main(commandArgs);
        } catch (Exception e) {
          System.err.println("Error during search: " + e.getMessage());
          e.printStackTrace();
          System.exit(1);
        }
        break;

      default:
        System.err.println("Error: Unknown command '" + command + "'");
        printUsage();
        System.exit(1);
    }
  }

  private static void printUsage() {
    System.out.println("Usage:");
    System.out.println("  java -jar ecir26.jar index --dataset <path> --index <path>");
    System.out.println("  java -jar ecir26.jar search [search arguments...]");
    System.out.println();
    System.out.println("Commands:");
    System.out.println("  index   - Index TREC documents using TRECIndexerLuceneRM");
    System.out.println("  search  - Search indexed documents using TRECSearcherLucene");
  }
}