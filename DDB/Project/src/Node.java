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
    public ArrayList<Receiver> receivers;
    private ArrayList<Sender> senders;
    private int[] sequence_numbers;
    private LSP my_last_lsp;
    private long last_lsp_sent_time;


    public Node(String line, int num_of_nodes){
        this.neighbors_dict = new Hashtable<Integer, Double[]>();
        this.neighbor2idx = new Hashtable<Integer, Integer>();
        this.num_of_nodes = num_of_nodes;
        this.senders = new ArrayList<Sender>();
        this.receivers = new ArrayList<Receiver>();
        String[] data = line.split(" ");
        this.id = Integer.parseInt(data[0]);
        init_adj_matrix(data);
        this.sequence_numbers = new int[this.num_of_nodes];
        for (int i = 0; i < this.num_of_nodes; i++) {
            sequence_numbers[i] = -1;
        }
    }
    public Node(Node node)
    {
        //copy constructor
        //used to rerun threads
        this.neighbors_dict = node.neighbors_dict;
        this.neighbor2idx = node.neighbor2idx;
        this.num_of_nodes = node.num_of_nodes;
        this.senders = node.senders;
        this.receivers = node.receivers;
        this.id = node.id;
        this.adj_matrix = node.adj_matrix;
        this.sequence_numbers = node.sequence_numbers;
        this.seq_num = node.seq_num;
        this.my_last_lsp = node.my_last_lsp;

    }


//            this.senders.add(new Sender(send_port));
//            this.receivers.add(new Receiver(listen_port));
//        System.out.println("Node " + this.id + " created, sender ports are: "+ this.senders.toString());

    public void print_graph(){
        //wait 1 seconds to make sure all the nodes are ready
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
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

    public void append_receiver(Receiver r){
        this.receivers.add(r);
    }
    public void append_sender(Sender s){
        this.senders.add(s);
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
        for (int j =0; j<1; j++){
            this.senders.parallelStream().forEach(sender ->{sender.send(message);});
//            for (int i = 0; i < this.senders.size(); i++) {
//                if (i != skip_idx || true) {
//                    this.senders.get(i).send(message);
//                    this.last_lsp_sent_time = System.currentTimeMillis();
//                }
//            }
        }
    }
    private void init_adj_matrix(String[] data) {
        //initializes the adjacency matrix of the node
        //sets all the weights to -1
        this.adj_matrix = new double[this.num_of_nodes][this.num_of_nodes];
        for (int i = 0; i < this.num_of_nodes; i++) {
            for (int j = 0; j < this.num_of_nodes; j++) {
                this.adj_matrix[i][j] = -1;
            }
        }
        for (int i = 1; i < data.length; i += 4) {
            int id2 = Integer.parseInt(data[i]);
            double weight = Double.parseDouble(data[i + 1]);
            int send_port = Integer.parseInt(data[i + 2]);
            int listen_port = Integer.parseInt(data[i + 3]);
            this.add_neighbor(send_port, listen_port, weight, id2);
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
        this.my_last_lsp = this.build_LSP();
        String message = this.my_last_lsp.toString();
        senders.forEach(sender -> sender.send(message));
        this.last_lsp_sent_time = System.currentTimeMillis();

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

    public void launch_ports() {
        //starts the listening and sending threads
        this.receivers.parallelStream().forEach(Receiver::bind);
        try {
            synchronized (this) {
                wait(128);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        ArrayList<Thread> threads = new ArrayList<Thread>();
        threads.addAll(this.senders);
        threads.addAll(this.receivers);

        threads.parallelStream().forEach(Thread::start);
    }

    public void halt_loop()
    {
        //stops the listening and sending threads
        this.senders.parallelStream().forEach(Sender::halt_loop);
        this.receivers.parallelStream().forEach(Receiver::halt_loop);
    }

    private boolean all_received(boolean[] received) {
        //checks if all the LSPs have been received
        for (int i = 0; i < received.length; i++) {
            if (!received[i]){
                return false;
            }
        }
        return true;
    }
    private String missing_messages(boolean[] received) {
        //returns a string of the missing LSPs
        String missing = "";
        for (int i = 0; i < received.length; i++) {
            if (!received[i]){
                missing += (i+1) + ", ";
            }
        }
        return missing;
    }
    @Override
    public void run() {
        //sets the sequence numbers for all the nodes in the graph to -1
        boolean[] received_message = new boolean[this.num_of_nodes];
        for (int i = 0; i < this.num_of_nodes; i++) {
            received_message[i] = false;
        }
        received_message[this.id-1] = true;
        this.start_broadcast();
        while (!this.all_received(received_message)) {
            List<String> messages = new ArrayList<>();
            synchronized (this) {
                this.receivers.parallelStream().forEachOrdered(r -> messages.add(r.returnStreamContent()));
            }
            for (int i = 0; i < this.neighbors_dict.size(); i++) {

                String message;
                message = messages.get(i);

                if (message != null) {
                    LSP lsp = new LSP(message);
                    this.update_adj_matrix(lsp);
                    if (lsp.get_seq_num() > this.sequence_numbers[lsp.get_source_id() - 1]) {
                        this.sequence_numbers[lsp.get_source_id() - 1] = lsp.get_seq_num();
                        received_message[lsp.get_source_id() - 1] = true;
                        this.send_message_to_all(message, i);
                    }
                }
            }
        }
    }
    public Node refreshed()
    {
        //returns a new node with the same parameters.
        //the reborn node could be run again and has the same parameters as the original node
        return new Node(this);
    }
    public void terminate(){
        //terminates the node
        this.receivers.parallelStream().forEach(Receiver::close);
        this.senders.parallelStream().forEach(Sender::close);
    }
}
