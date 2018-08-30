/*
INF 552 Homework 1
Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
Date: 2/7/2018
 */

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Node {

    private String root;
    private HashMap<String, int[]> yn_table;
    private HashMap<String, ArrayList<Integer>> sub_rows;
    private HashMap<String, Node> children;
    private int total_row_num = 0;

    private double entropy_in;
    private double entropy_out;
    private double IG;
    private HashMap<String, Double>  sub_entropy;



    public Node()
    {
        yn_table = new HashMap<>();
    }


    public Node(String att)
    {
        root = att;
        yn_table = new HashMap<>();
        sub_rows = new HashMap<>();
    }

    public Node(String train_data[][], ArrayList<Integer> rows, String att)
    {
        root = att;
        String att_values[] = main.ATTRIBUTES_MAP.get(att);
        yn_table = new HashMap<>();
        sub_rows = new HashMap<>();
        children = new HashMap<>();

        sub_entropy = new HashMap<>();

        for(String att_value : att_values)
        {
            yn_table.put(att_value, new int[2]);
            sub_rows.put(att_value, new ArrayList<Integer>());
            sub_entropy.put(att_value, 0.0);
        }

        build(train_data, rows);
    }


    public void build(String train_data[][], ArrayList<Integer> rows)
    {
        int column = main.ATTRIBUTE_NAMES.indexOf(root);
        for(int i = 0; i < rows.size(); i ++)
        {
            int index = rows.get(i);
            String key = train_data[index][column];
            sub_rows.get(key).add(index);
            build_yn(key, train_data[index][main.ATTRIBUTES_NUMBER - 1]);
        }
    }


    public void build_yn(String key, String yes_or_no)
    {
        int pre[] = yn_table.get(key);
        if(yes_or_no.equals("Yes"))
        {
            pre[0] ++;
            yn_table.put(key, pre);
            total_row_num ++;
        }
        else if(yes_or_no.equals("No"))
        {
            pre[1] ++;
            yn_table.put(key, pre);
            total_row_num ++;
        }
        else
        {
            System.out.println("ERROR in yn_table");
        }
    }


    public double getEntropyOut()
    {
        double entropy_out = 0;
        for(Map.Entry<String, int[]> att_entry : yn_table.entrySet())
        {
            if(total_row_num == 0)
            {
                System.out.println("total_row_num == 0 ! Something Wrong!" );
                return entropy_out;
            }
            double sub_entrpoy_value = calculateEntropy(att_entry.getValue());
            sub_entropy.put(att_entry.getKey(), sub_entrpoy_value);
            entropy_out += (double)((att_entry.getValue()[0] + att_entry.getValue()[1])) / total_row_num * sub_entrpoy_value;
        }
        return entropy_out;
    }


    public double calculateEntropy(int yes_and_no[]){

        if(yes_and_no[0] == 0 || yes_and_no[1] == 0)
        {
            return 0;
        }

        double sum = yes_and_no[0] + yes_and_no[1];

        double entropy = (-1) * (Math.log(yes_and_no[0] / sum) /  Math.log(2.0) * (yes_and_no[0] / sum) +
                Math.log(yes_and_no[1] / sum) /  Math.log(2.0) * (yes_and_no[1] / sum));
        return entropy;
    }

    public double getIG(double entropy_in) {
        this.entropy_in = entropy_in;
        this.entropy_out = getEntropyOut();
        IG = this.entropy_in - this.entropy_out;
        return IG;
    }

    public String getRootName(){
        return root;
    }

    public ArrayList<Integer> getSub_Rows(String att_value){
        return new ArrayList<>(sub_rows.get(att_value));
    }

    public void addChild(String att_value, Node new_child){
        children.put(att_value, new_child);
    }

    public double calculateEntropy(String att_value){
        return calculateEntropy(yn_table.get(att_value));
    }

    public void print(){
        String att[] = main.ATTRIBUTES_MAP.get(root);
        for(int i = 0; i < att.length; i++)
        {
            System.out.print(att[i]);
            System.out.print("  rows=" + Arrays.toString(sub_rows.get(att[i]).toArray()));
            System.out.print("  enjoy=" + Arrays.toString(yn_table.get(att[i])));
            System.out.printf("  en_in=%.3f" , entropy_in);
            System.out.printf("  sub_en_out=%.3f" , sub_entropy.get(att[i]));
            System.out.println("  weight=" + (yn_table.get(att[i])[0]+ yn_table.get(att[i])[1]) + "/" + total_row_num);
        }
        System.out.printf("IG = %.3f\n" ,IG);
    }


    // Added after first grading. 02/27/2018
    public String makePrediction(HashMap<String, String> test_data){
        Node child_node = children.get(test_data.get(root));
        System.out.print("Checking " + root + ", value = " + test_data.get(root) + ".\t");
        if(child_node.root.equals("yes"))
        {
            System.out.println("Got result: Yes");
            return "Yes";
        }
        else if(child_node.root.equals("no"))
        {
            System.out.println("Got result: No");
            return "No";
        }
        else if(child_node.root.equals("empty"))
        {
            System.out.println("Got result: Empty");
            return "Can not make a prediction";
        }
        else
        {
            System.out.println("Next check: " + child_node.root);
            return child_node.makePrediction(test_data);
        }
    }
}
