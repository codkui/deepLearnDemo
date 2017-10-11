import SimpleHTTPServer
from BaseHTTPServer import HTTPServer,BaseHTTPRequestHandler
import SocketServer
import json
import numpy as np
PORT = 8003


class TestHTTPHandle(BaseHTTPRequestHandler): 
    def do_GET(self):
        buf = "It works"
        self.protocal_version = "HTTP/1.1"
  
        self.send_response(200)
  
        self.send_header("Welcome", "Contect")     
  
        self.end_headers()
        
        if(self.path=="/favicon.ico"):
            self.wfile.write(buf)
            return
        print("收到请求")
        
        meg=self.path[1:]
        req=json.loads(meg)
        req=np.array([req])
        print(req)
        
        
        self.wfile.write(buf)
 
httpd = SocketServer.TCPServer(("", PORT), TestHTTPHandle)
 
print "serving at port", PORT
httpd.serve_forever()