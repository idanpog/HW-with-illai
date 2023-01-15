import java.net.*;
import java.util.*;
import java.io.*;

public class Node extends Thread {
    //the base class for all nodes
    private int id;
    private Hashtable<Integer, Double[]> neighbours_dict;
    private int[][] adj_matrix;

    public Node(int id){
        this.id = id;
        this.neighbours = new utils.hash;

    }
    public void print_graph(){
        //prints the adjacency matrix of the node as shown in the output examples
        System.out.println(adj_matrix);
    }
    public void update_neighbour_weight(int id, double weight){
        //updates the weight of the edge between this node and the node with id = id
        this.neighbours_dict.get(id)[2] = weight;
    }
    public void add_neighbour(double send_port, double listen_port, double weight, int id){
        double[] new_neighbour = {send_port, listen_port, weight};
        this.neighbours_dict.put(id, new_neighbour);
    }
    public void update_adj_matrix(){
        //updates the adjacency matrix of the node
        for (int i = 0; i < this.neighbours_dict.size(); i++) {
            for (int j = 0; j < this.neighbours_dict.size(); j++) {
                if(i == j){
                    this.adj_matrix[i][j] = 0;
                }
                else{
                    this.adj_matrix[i][j] = -1;
                }
            }
        }
        for (int i = 0; i < this.neighbours_dict.size(); i++) {
            for (int j = 0; j < this.neighbours_dict.size(); j++) {
                if(i != j){
                    this.adj_matrix[i][j] = this.neighbours_dict.get(i)[2];
                }
            }
        }
    }

}
