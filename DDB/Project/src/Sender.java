import java.net.*;
import java.io.*;

public class Sender extends Thread {
    private int port;
    private DataOutputStream outputStream;
    private Socket socket;
    private boolean alive;
    private java.util.concurrent.BlockingQueue<String> message_queue;

    public Sender(int port)
    {
        this.port = port;
        this.message_queue = new java.util.concurrent.LinkedBlockingQueue<String>();
    }
@Override
    public void run()
    {
        //System.out.println("Sender on port " + port + " started");
        this.alive = true;
        int count = 0;
        while(count < 10) {
            try {
//                Thread.sleep(32);
                assert this.port != -1;
//                System.out.println("Sender trying to connect to port " + port);
                this.socket = new Socket("localhost", this.port);
//                System.out.println("Sender connected to port " + this.port);
                this.outputStream = new DataOutputStream(socket.getOutputStream());
//                System.out.println("Sender started an output stream on port " + port);
                break;

            } catch (IOException e) {
                System.out.println("Error in Sender with port " + this.port+ ": " + e);
                System.out.println("Trying again...");
            }
            count++;
            try {
                synchronized(this){
                wait(10);}
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        if (count==10)
        {System.out.println("Failed to start Sender on port " + port);}
        else
        {
            synchronized (this) { start_pooling_loop();}
        }
        //System.out.println("Sender on port " + port + " terminated");
    }

    public void send(String message)
    {
        synchronized (this.message_queue) {
            try {
                this.message_queue.put(message);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    private void internal_send(String message)
    {

        synchronized (this.outputStream) {
            try {
                // Send the message
                outputStream.writeUTF(message);
                outputStream.flush();
//                System.out.println("Sender sent a message on port " + port);
            } catch (IOException e) {
                System.out.println("Error: " + e);
            }
        }
    }

    public void start_pooling_loop()
    {
//        System.out.println("Sender started pooling loop on port " + port);
        while(this.alive)
        {
            try {

                String message = this.message_queue.take();
                if (message.equals("terminate"))
                {
                    this.alive = false;
                    break;
                }
                this.internal_send(message);

            } catch (InterruptedException e) {
                if (this.alive)
                {
                    e.printStackTrace();
                }
            }
        }
    }

    public void halt_loop()
    {
        this.alive = false;
        this.send("terminate");
    }

    public void close()
    {
        try {
            // Close the socket
            outputStream.close();
            socket.close();
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }

    }
    public String toString(){
        return "Sender on port " + this.port;
    }
}

