import java.net.*;
import java.io.*;

public class Receiver extends Thread {
    private int port = 0;
    private Socket socket = null;
    private ServerSocket serverSocket = null;
    private DataInputStream inputStream = null;
    private java.util.concurrent.BlockingQueue<String> message_queue;
    private boolean alive;

    public Receiver(int port) {
        this.port = port;
        this.message_queue = new java.util.concurrent.LinkedBlockingQueue<String>();
    }
@Override
    public void run() {
        /*
    accepts the sockets from the matching senders, and starts the pooling loop
    note that this method should be called only after bind has been called.
    */
        this.alive = true;
        //System.out.println("Receiver on port " + port + " started");
        try {

            this.socket = this.serverSocket.accept();
            this.inputStream = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
        } catch (IOException e) {
            System.out.println("Error launching the receiver: " + e);
        }
        start_pooling_loop();
    }
    public void start_pooling_loop()
    {
        /*
        * starts the pooling loop, constantly seeks for messages and puts them in the message queue upon arrival
        * */
        while (this.alive) {
            try {
//                System.out.println("Receiver waiting for a message on port " + port);
                String message = this.inputStream.readUTF();
                if (message.equals("terminate"))
                {
                    this.alive = false;
                    break;
                }
                synchronized (this.message_queue) { this.message_queue.put(message);}
                }
            catch (IOException e) {
                if (this.alive) {
                    System.out.println("Error in Receiver with port " + this.port + ": " + e);
                    System.out.println("Trying again...");
                }
            }
            catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }


    public void bind(){
        try {
            this.serverSocket = new ServerSocket();
            serverSocket.bind(new InetSocketAddress("localhost", port));
//            System.out.println("Receiver bound to port " + port);

        } catch (IOException e) {
            System.out.println("Error binding the receiver: " + e);
        }
    }

    public String returnStreamContent() {
        // return a message sitting in the queue, null if the queue is empty, thread safe
        String current_message = null;
        synchronized (this.message_queue) {
            if (!this.message_queue.isEmpty()) {
                try {
                    current_message = this.message_queue.take();
                } catch (InterruptedException e) {
                    if (this.alive) {
                        e.printStackTrace();
                    }

                }
            }
        }
        return current_message;
    }

    private int count_occurrences(String big, String small) {
        int count = 0;
        int index = 0;
        while (index != -1) {
            index = big.indexOf(small, index);
            if (index != -1) {
                count++;
                index += small.length();
            }
        }
        return count;
    }

    public void halt_loop()
    {
        this.alive = false;
    }

    public void close()
    {
        try {
            // Close the socket
            inputStream.close();
            socket.close();
            serverSocket.close();
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
    }

}
