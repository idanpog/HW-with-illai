import java.io.*;

import java.net.*;
import java.util.Arrays;
import java.util.Queue;
import java.util.concurrent.*;
public class NodeClient extends Thread{
    private final int port; //port number
    private final int id; //id of the node
    private final int targetid; //id of the node that will receive the message
    private final Boolean[] listened; //list of nodes that have been listened to
    private final LinkedBlockingQueue<Pair<Pair<Integer, Integer>, Double>[]> queue = new LinkedBlockingQueue<>();
    public NodeClient(int port,int id, int targetid, Pair<Pair<Integer, Integer>, Double>[] linkState, Boolean[] listened) throws InterruptedException
    {
        this.port = port;
        this.id=id;
        this.targetid = targetid;
        this.listened = listened;
        queue.put(linkState);
    }
    @Override
    public void run() {
        super.run();
        boolean connected = false;
        while(!connected)
        {
            try(Socket sock = new Socket("127.0.0.1", this.port); //create a socket
                ObjectOutputStream out =
                        new ObjectOutputStream(sock.getOutputStream()); //create an output stream
            )
            {
                connected = true;
                while(sock.isConnected()) //while the socket is connected
                        {
                            Pair<Pair<Integer, Integer>, Double>[] linkState = queue.poll(); //get the link state
                            if(linkState!=null)
                            {
                        out.writeObject(linkState);
                        out.flush();
                    }
                    synchronized(this.listened) // synchronize the list of nodes that have been listened to
                    {
                        if (queue.size()==0){
                            boolean all = true;
                            for(int i=0;i<this.listened.length;i++)
                            {
                                if(i+1!=this.id && !this.listened[i]) {
                                    all = false;
                                    break;
                                }
                            }
                            if(all)
                            {
                                return;
                            }
                        }
                    }
                }
            } catch (IOException e) {

            }
        }
    }
    public int get_id()
    {
        return this.id;
    }
    public synchronized void queueMessage(Pair<Pair<Integer, Integer>, Double>[] linkState) throws InterruptedException
    {
        this.queue.put(linkState); //add the link state to the queue
    }
}
