import java.io.*;
import java.util.*;
import java.net.*;
import java.util.concurrent.CountDownLatch;

public class Node implements Runnable{
    private int id; //id of the node
    private int numNodes; //number of nodes in the network
    private Double[][] matrix; //matrix of the node
    private Pair<Integer, Integer>[] ports; //list of ports that will be used to listen to other nodes
    private Boolean[] listened; //list of nodes that have been listened to

    private int numNeighbors; //number of neighbors of the node

    private NodeClient[] nodeClients; //list of clients that will be used to send messages to other nodes
    private CountDownLatch endSignal; //signal that will be used to wait for all threads to finish
    private NodeServer[] listeningServers; //list of servers that will be used to listen to other nodes

    /**
     * @param id the id of the node
     * @param numNodes the number of nodes in the network
     * @param endSignal the signal that will be used to wait for all threads to finish
     */
    public Node(int id, int numNodes, CountDownLatch endSignal){
        this.id = id;
        this.numNodes = numNodes;
        this.endSignal = endSignal;
        this.matrix = new Double[numNodes][numNodes];
        for (int i = 0; i < numNodes; i++){
            for (int j = 0; j < numNodes; j++){
                this.matrix[i][j] = -1.0;
            }}
        this.ports = new Pair[numNodes];
        this.numNeighbors = 0;
    }

    @Override
    public void run() {
        Pair<Pair<Integer, Integer>, Double>[] linkState = new Pair[this.numNeighbors];
        int index = 0;
        this.listened = new Boolean[this.numNodes]; //list of nodes that have been listened to
        for (int i = 0; i < this.numNodes; i++){
            this.listened[i] = false;
        }
        this.listened[this.id-1] = true;
        for (int i=1; i<this.numNodes+1; i++)
        {
            if(this.ports[i-1]!=null)
            {
                linkState[index] = new Pair<>(new Pair<>(this.id, i), this.matrix[this.id-1][i-1]);
                index++;
            }
        }
        nodeClients = new NodeClient[numNeighbors];
        index = 0;
        for (int i=0;i<this.numNodes; i++)
        {
            if(this.ports[i]!=null)
            {
                try { //create a client socket
                    nodeClients[index] = new NodeClient(this.ports[i].getKey(),this.id,i+1, linkState, this.listened);
                    nodeClients[index].start();
                    index++;
                } catch (InterruptedException e) {
                    throw new RuntimeException(e); //throw an exception if the thread is interrupted
                }
            }
        }
        listeningServers = new NodeServer[this.numNodes];
        for(int i=0; i<this.numNodes; i++)
        {
            if(ports[i]!=null)
            {
                listeningServers[i] = new NodeServer(this.ports[i].getValue(), this.listened, this.matrix, nodeClients
                        , this.id, i+1); //create a server socket
                listeningServers[i].start(); //start the server
            }
        }
        this.terminate();
        this.endSignal.countDown();
    }

    public void addNeighbor(int j, double weight, int sendPort, int listenPort)
    {
        this.matrix[this.id-1][j-1] = weight;
        this.matrix[j-1][this.id-1] = weight;
        this.ports[j-1] = new Pair<>(sendPort, listenPort);
        this.numNeighbors+=1;
    }
    public void updateWeight(int j, double weight)
    {
        this.matrix[this.id-1][j-1] = weight;
        this.matrix[j-1][this.id-1] = weight;
    }

    public void print_graph() {
        for(int i=0; i<numNodes; i++){
            for(int j=0; j<numNodes; j++){
                System.out.print(this.matrix[i][j]);
                if (j != numNodes-1){
                    System.out.print(", ");
                }
            }
            System.out.println();
        }
    }

    public void setLatch(CountDownLatch endSignal) {
        this.endSignal = endSignal; //set the latch
    }
    public void terminate(){
        for(int i=0;i<this.numNeighbors;i++)
        {
            try {
                nodeClients[i].join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        for(int i=0;i<this.numNodes;i++) {
            if (this.ports[i] != null) {
                try {
                    listeningServers[i].join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
