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
//            System.out.println("Receiver accepted a client on port " + port);
            this.socket.setSoTimeout(1024);

            BufferedInputStream buff = new BufferedInputStream(socket.getInputStream());
            this.inputStream = new DataInputStream(buff);
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
        try {
            String message = inputStream.readUTF();

            return message;
        }
        catch (SocketTimeoutException e) {
            return null;
        }
        catch (IOException e) {
            System.out.println("Error: " + e);
        }
        return null;
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
