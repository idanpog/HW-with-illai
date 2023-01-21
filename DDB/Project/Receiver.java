import java.net.*;
import java.io.*;

public class Receiver extends Thread {
    private int port = 0;
    private Socket socket = null;
    private ServerSocket serverSocket = null;
    private DataInputStream inputStream = null;

    public Receiver(int port)
    {
        this.port = port;
    }

    public void start()
    {/*
    accepts the sockets from the matching senders,
    note that this method should be called only after bind has been called.
    */
        try {
//            System.out.println("Receiver waiting on port " + port);
            this.socket = this.serverSocket.accept();
            this.inputStream = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
//            System.out.println("Receiver accepted a client on port " + port);
            //this.socket.setSoTimeout(16);

//            System.out.println("Receiver started on port " + port);

        } catch (IOException e) {
            System.out.println("Error launching the receiver: " + e);
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
        // returns the content of the input stream, timeout 10ms return null otherwise
        String message = null;
        synchronized (this) {
            try {
                if (inputStream.available() > 0) {
                    message = inputStream.readUTF();
                    return message;

                } else {
                    return null;
                }
            } catch (SocketTimeoutException e) {
                return null;
            } catch (IOException e) {
                System.out.println("Error: " + e);
            }
            return null;
        }
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
