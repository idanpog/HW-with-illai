import java.net.*;
import java.util.*;
import java.io.*;

public class Node extends Thread {
    //the base class for all nodes
    private int id;
    private int seq_num =0;
    private double[][] adj_matrix;
    private int num_of_nodes;
    private Hashtable<Integer, Double[]> neighbors_dict;
    private Hashtable<Integer, Integer> neighbor2idx;
    private ArrayList<Receiver> receivers;
    private ArrayList<Sender> senders;


    public Node(String line, int num_of_nodes){
        this.neighbors_dict = new Hashtable<Integer, Double[]>();
        this.neighbor2idx = new Hashtable<Integer, Integer>();
        this.num_of_nodes = num_of_nodes;
        this.senders = new ArrayList<Sender>();
        this.receivers = new ArrayList<Receiver>();
        String[] data = line.split(" ");
        this.id = Integer.parseInt(data[0]);
        init_adj_matrix();
        for (int i = 1; i < data.length; i+=4) {
            int id2 = Integer.parseInt(data[i]);
            double weight = Double.parseDouble(data[i+1]);
            int send_port = Integer.parseInt(data[i+2]);
            int listen_port = Integer.parseInt(data[i+3]);
            this.add_neighbor(send_port, listen_port, weight, id2);
            this.senders.add(new Sender(send_port));
            this.receivers.add(new Receiver(listen_port));
        }
//        System.out.println("Node " + this.id + " created, sender ports are: "+ this.senders.toString());
    }
    public void print_graph(){
        //prints the adjacency matrix of the node as shown in the output examples
        for (int i = 0; i < this.num_of_nodes; i++) {
            for (int j = 0; j < this.num_of_nodes; j++) {
                System.out.print(this.adj_matrix[i][j]);
                if (j != this.num_of_nodes-1){
                    System.out.print(", ");
                }
            }
            System.out.println();
        }
    }
    public void update_neighbor_weight(int id, double weight){
        //updates the weight of the edge between this node and the node with id = id
        this.adj_matrix[this.id-1][id-1] = weight;
        this.adj_matrix[id-1][this.id-1] = weight;
    }
    public void add_neighbor(double send_port, double listen_port, double weight, int id){
        //adds a neighbor to the node
        this.neighbors_dict.put(id, new Double[]{send_port, listen_port});
        this.adj_matrix[this.id-1][id-1] = weight;
        this.adj_matrix[id-1][this.id-1] = weight;
    }
    void send_message_to_all(String message, Integer skip_idx) {
        // Iterate through all routers in the network
        for (int i = 0; i < this.senders.size(); i++) {
            if (i != skip_idx) {
                this.senders.get(i).send(message);
            }
        }
    }
    private void init_adj_matrix(){
        //initializes the adjacency matrix of the node
        //sets all the weights to -1
        this.adj_matrix = new double[this.num_of_nodes][this.num_of_nodes];
        for (int i = 0; i < this.num_of_nodes; i++) {
            for (int j = 0; j < this.num_of_nodes; j++) {
                this.adj_matrix[i][j] = -1;
            }
        }
    }
    public LSP build_LSP(){
        //returns the LSP of the node
        Set<Pair<Pair<Integer, Integer>, Double>> l_v = new HashSet<Pair<Pair<Integer, Integer>, Double>>();
        this.neighbors_dict.forEach((key, value) -> {
            double weight = this.adj_matrix[this.id-1][key-1];
            Pair<Integer, Integer> edge = new Pair<Integer, Integer>(this.id, key);
            l_v.add(new Pair<Pair<Integer, Integer>, Double>(edge, weight));
        });
        LSP lsp = new LSP(this.seq_num, this.id, l_v);
        seq_num++;
        return lsp;
    }
    void start_broadcast() {
        // Broadcast LSP to all neighbors
        LSP lsp = this.build_LSP();
        String message = lsp.toString();
        senders.parallelStream().forEach(sender -> sender.send(message));
    }

    public void update_adj_matrix(LSP lsp) {
        // Update the adjacency matrix based on the LSP
        Set<Pair<Pair<Integer, Integer>, Double>> l_v = lsp.get_lv();
        l_v.forEach(pair -> {
            Pair<Integer, Integer> edge = pair.getKey();
            int id1 = edge.getKey();
            int id2 = edge.getValue();
            double weight = pair.getValue();
            this.adj_matrix[id1-1][id2-1] = weight;
            this.adj_matrix[id2-1][id1-1] = weight;
        });
    }

    public void launch_ports(){
        //starts the listening and sending threads
        ArrayList<Thread> threads = new ArrayList<Thread>();
        threads.addAll(this.senders);
        threads.addAll(this.receivers);
        threads.parallelStream().forEach(Thread::start);
        //joins the threads
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }


    public void run_link_state(){
        //sets the sequence numbers for all the nodes in the graph to -1
        int[] sequence_numbers = new int[this.num_of_nodes];
        for (int i = 0; i < this.num_of_nodes; i++) {
            sequence_numbers[i] = -1;
        }

        this.start_broadcast();
        boolean changed = true;
        while (changed)
        {
            changed = false;
            for (int i = 0; i < this.neighbors_dict.size(); i++) {
                String message = this.receivers.get(i).returnStreamContent();
                if (message != null) {
                    changed = true;
                    LSP lsp = new LSP(message);
                    this.update_adj_matrix(lsp);
                    if (lsp.get_seq_num() > sequence_numbers[lsp.get_source_id() - 1]) {
                        sequence_numbers[lsp.get_source_id() - 1] = lsp.get_seq_num();
                        this.send_message_to_all(message, i);
                    }
                }
            }
        }
    }

    public void terminate(){
        //terminates the node
        this.receivers.parallelStream().forEach(Receiver::close);
        this.senders.parallelStream().forEach(Sender::close);
    }
}
