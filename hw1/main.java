/*
INF 552 Homework 1
Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
Date: 2/7/2018
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.Scanner;
import java.lang.Math;
import java.util.HashMap;

public class main {

    final public static int ATTRIBUTES_NUMBER = 7;
    final public static ArrayList<String>  ATTRIBUTE_NAMES = new ArrayList<>
            (Arrays.asList("Occupied", "Price", "Music", "Location", "VIP", "Favorite Beer"));
    final private static String INPUT_FILE_NAME = "dt-data.txt";

    public static HashMap<String, String[]> ATTRIBUTES_MAP = new HashMap<>();

    final public static String[] Occupied = {"High", "Moderate", "Low"};
    final public static String[] Price = {"Expensive", "Normal", "Cheap"};
    final public static String[] Music = {"Loud", "Quiet"};
    final public static String[] Location = {"Talpiot", "City-Center", "Mahane-Yehuda", "Ein-Karem", "German-Colony"};
    final public static String[] VIP = {"Yes", "No"};
    final public static String[] Favorite_Beer = {"Yes", "No"};
    final public static String[] Enjoy = {"Yes", "No"};

    private static String train_data[][];

    private static String[] test_data_array = {"Moderate", "Cheap", "Loud", "City-Center", "No", "No"};
    private static HashMap<String, String> test_data = new HashMap<>();


    public static void main(String[] args){

        String filename = new String();
        if (args.length < 1) {
            System.out.println("ERROR: missing file name command line argument");
        }
        else {
            filename = args[0];
        }

        filename = INPUT_FILE_NAME;
        try{
            constructAttTable();
            train_data = readFile(filename);
            int row_num = 0;
            for (String row[]  : train_data)
            {
                row_num ++;
            }

            ArrayList<Integer> rows = new ArrayList<>();
            for(int i = 0; i < row_num; i ++)
            {
                rows.add(i);
            }

            int yes_and_no[] = countYesAndNo(rows);
            double entropy = calculateEntropy(yes_and_no);
            int level = 1;
            Node root = ID3(rows, ATTRIBUTE_NAMES, entropy, level);
            constructTestData();
            System.out.println("\nRunning status:");
            System.out.println("\nOur prediction for the given test data is:\n" + root.makePrediction(test_data));
//            root.print;
        }
        catch (IOException exception)
        {
            System.out.println("File not found !!");
        }
    }


    private static String[][] readFile(String filename) throws IOException {

        File inFile = new File(filename);
        try(Scanner in = new Scanner(inFile))
        {
            in.nextLine();
            in.nextLine();

            ArrayList<String> everyLine = new ArrayList<>();
            while(in.hasNextLine()){
                everyLine.add(in.nextLine());
            }

            String train_data[][] = new String[everyLine.size()][ATTRIBUTES_NUMBER];

            for(int i = 0; i < everyLine.size(); i ++)
            {
                String line = everyLine.get(i);
                Scanner lineScanner = new Scanner(line);
                lineScanner.next();
                String temp;
                for(int j = 0; j < ATTRIBUTES_NUMBER; j ++ )
                {
                    temp = lineScanner.next();
                    train_data[i][j] = temp.substring(0, temp.length() - 1);
                }
            }

            return train_data;
        }
    }


    public static Node ID3(ArrayList<Integer> rows, ArrayList<String> atts, double entropy, int i){

        Node root = new Node();
        int yes_and_no[] = countYesAndNo(rows);

        String tab = "";
        for(int k = 2; k <= i; k++ )
        {
            tab += "   ";
        }


        if(yes_and_no[0] == rows.size())
        {
            System.out.println("YES");
            return new Node("yes");
        }
        if(yes_and_no[1] == rows.size())
        {
            System.out.println("NO");
            return new Node("no");
        }
        if(atts.isEmpty())
        {
            System.out.println("(TIE)");
            if(yes_and_no[0] >= yes_and_no[1])
            {
                return new Node("yes");
            }
            else
            {
                return new Node("no");
            }
        }

        root = bestAttribute(rows, atts, entropy);
        tab += " ";

        System.out.println(root.getRootName());
        for(String att_value : ATTRIBUTES_MAP.get(root.getRootName()))
        {
            System.out.print(tab + att_value + ": ");
            ArrayList<Integer> sub_rows = root.getSub_Rows(att_value);
            if(sub_rows.isEmpty()) {
                System.out.println("(EMPTY)");
                root.addChild(att_value, new Node("empty"));
            }
            else {
                ArrayList<String> atts_left = new ArrayList<>(atts);
                atts_left.remove(root.getRootName());
                root.addChild(att_value, ID3(sub_rows, atts_left, root.calculateEntropy(att_value), i+1));
            }
        }
        return root;
    }


    private static Node bestAttribute(ArrayList<Integer> rows, ArrayList<String> atts, double entropy_in) {

        Node best_table = new  Node(train_data, rows, atts.get(0));

        if(atts.size() > 1)
        {
            for(int i = 1; i < atts.size(); i ++)
            {
                Node curr_table = new Node(train_data, rows, atts.get(i));

                // Strictly greater than. Because if they are equal, we should be using the one with smaller index.
                if(curr_table.getIG(entropy_in) > best_table.getIG(entropy_in))
                {
                    best_table = curr_table;

                }
            }
        }
        return best_table;
    }


    public static int[] countYesAndNo(ArrayList<Integer> rows){
        int yes_and_no[] = new int[2];
        for(int i = 0; i < rows.size(); i ++)
        {
            if(train_data[rows.get(i)][ATTRIBUTES_NUMBER - 1].equals("Yes"))
            {
                yes_and_no[0] ++;
            }
            else if(train_data[rows.get(i)][ATTRIBUTES_NUMBER - 1].equals("No"))
            {
                yes_and_no[1] ++;
            }
            else
            {
                System.out.print(train_data[rows.get(i)][ATTRIBUTES_NUMBER - 1]);
            }
        }
        return yes_and_no;
    }


    public static double calculateEntropy(int yes_and_no[]){
        double sum = yes_and_no[0] + yes_and_no[1];
        double entropy = (-1) * (Math.log(yes_and_no[0] / sum) /  Math.log(2.0) * (yes_and_no[0] / sum) +
                            Math.log(yes_and_no[1] / sum) /  Math.log(2.0) * (yes_and_no[1] / sum));
        return entropy;
    }


    private static void constructAttTable(){
        ATTRIBUTES_MAP.put("Occupied", Occupied);
        ATTRIBUTES_MAP.put("Price", Price);
        ATTRIBUTES_MAP.put("Music", Music);
        ATTRIBUTES_MAP.put("Location", Location);
        ATTRIBUTES_MAP.put("VIP", VIP);
        ATTRIBUTES_MAP.put("Favorite Beer", Favorite_Beer);
    }


    // Added after first grading. 02/27/2018
    private static void constructTestData(){
        String[] test_data_for_print = new String[ATTRIBUTE_NAMES.size()];
        for(int i = 0; i < ATTRIBUTE_NAMES.size(); i ++)
        {
            test_data.put(ATTRIBUTE_NAMES.get(i), test_data_array[i]);
            test_data_for_print[i] = ATTRIBUTE_NAMES.get(i) + "=" + test_data_array[i];
        }
        System.out.println("\n\nThe given test data is:\n" + Arrays.toString(test_data_for_print));
    }
}

