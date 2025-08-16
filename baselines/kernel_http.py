#!/usr/bin/env python
"""
This script starts a simple HTTP server that can execute Python code
and maintain state between requests. It can also optionally start an IPython kernel
for direct kernel connections.
"""

import os
import sys
import json
import time
import signal
import socket
import traceback
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import argparse
import threading

# Optional IPython kernel support
try:
    from ipykernel.kernelapp import IPKernelApp
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False

# Global state dictionary to store variables
global_state = {}
kernel_app = None
kernel_thread = None

class CodeExecutor:
    """Class to execute Python code and capture output."""
    
    def __init__(self):
        self.globals = {'__builtins__': __builtins__}
        
    def execute(self, code):
        """Execute Python code and return the output."""
        stdout = StringIO()
        stderr = StringIO()
        
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, self.globals)
            output = stdout.getvalue()
            error = stderr.getvalue()
            return {
                "status": "success",
                "output": output,
                "error": error
            }
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return {
                "status": "error",
                "output": stdout.getvalue(),
                "error": error
            }
    
    def get_variables(self):
        """Get a list of defined variables."""
        import inspect
        import builtins
        
        variables = {}
        for name, value in self.globals.items():
            # Skip builtins, modules, and private variables
            if (not name.startswith('_') and 
                name not in dir(builtins) and 
                not inspect.ismodule(value) and
                not inspect.isfunction(value) and
                not inspect.isclass(value)):
                variables[name] = type(value).__name__
        
        return variables
    
    def get_variable(self, name):
        """Get the value of a specific variable."""
        import inspect
        import builtins
        
        if (name in self.globals and 
            not name.startswith('_') and 
            name not in dir(builtins) and 
            not inspect.ismodule(self.globals[name]) and
            not inspect.isfunction(self.globals[name]) and
            not inspect.isclass(self.globals[name])):
            
            try:
                # Try to convert to JSON-serializable format
                import pandas as pd
                import numpy as np
                
                value = self.globals[name]
                
                # Handle pandas DataFrame
                if isinstance(value, pd.DataFrame):
                    return value.to_dict(orient='records')
                
                # Handle numpy arrays
                if isinstance(value, np.ndarray):
                    return value.tolist()
                
                # Handle other numpy types
                if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                     np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(value)
                
                if isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                    return float(value)
                
                if isinstance(value, (np.bool_)):
                    return bool(value)
                
                # Try to convert to string for other types
                return str(value)
            except:
                return str(self.globals[name])
        else:
            return f"Variable '{name}' not found or not accessible"

# Create a code executor
executor = CodeExecutor()

class CodeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for code execution."""
    
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(post_data)
            
            if self.path == '/execute':
                if 'code' in data:
                    result = executor.execute(data['code'])
                else:
                    result = {"status": "error", "error": "No code provided"}
            
            elif self.path == '/variables':
                result = {"status": "success", "variables": executor.get_variables()}
            
            elif self.path == '/variable':
                if 'name' in data:
                    result = {
                        "status": "success", 
                        "name": data['name'], 
                        "value": executor.get_variable(data['name'])
                    }
                else:
                    result = {"status": "error", "error": "No variable name provided"}
            
            elif self.path == '/upload_file':
                if 'filename' in data and 'content' in data:
                    try:
                        # Decode the base64 content
                        file_content = base64.b64decode(data['content'])
                        
                        # Create directories if they don't exist
                        file_path = data['filename']
                        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                        
                        # Write the file
                        with open(file_path, 'wb') as f:
                            f.write(file_content)
                            
                        result = {"status": "success", "message": f"File uploaded to {file_path}"}
                    except Exception as e:
                        result = {"status": "error", "error": f"Error uploading file: {str(e)}"}
                else:
                    result = {"status": "error", "error": "Filename or content not provided"}
            
            elif self.path == '/download_file':
                if 'filename' in data:
                    try:
                        file_path = data['filename']
                        if os.path.exists(file_path):
                            # Read the file
                            with open(file_path, 'rb') as f:
                                file_content = f.read()
                                
                            # Encode the content as base64
                            encoded_content = base64.b64encode(file_content).decode('utf-8')
                            
                            result = {
                                "status": "success", 
                                "filename": file_path,
                                "content": encoded_content
                            }
                        else:
                            result = {"status": "error", "error": f"File not found: {file_path}"}
                    except Exception as e:
                        result = {"status": "error", "error": f"Error downloading file: {str(e)}"}
                else:
                    result = {"status": "error", "error": "Filename not provided"}
            
            elif self.path == '/list_files':
                if 'directory' in data:
                    try:
                        directory = data['directory']
                        if os.path.exists(directory) and os.path.isdir(directory):
                            files = os.listdir(directory)
                            result = {"status": "success", "directory": directory, "files": files}
                        else:
                            result = {"status": "error", "error": f"Directory not found: {directory}"}
                    except Exception as e:
                        result = {"status": "error", "error": f"Error listing files: {str(e)}"}
                else:
                    result = {"status": "error", "error": "Directory not provided"}
            
            else:
                result = {"status": "error", "error": f"Unknown endpoint: {self.path}"}
            
            self._set_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            
        except json.JSONDecodeError:
            self._set_headers()
            self.wfile.write(json.dumps({"status": "error", "error": "Invalid JSON"}).encode('utf-8'))
        except Exception as e:
            self._set_headers()
            self.wfile.write(json.dumps({
                "status": "error", 
                "error": f"Server error: {str(e)}"
            }).encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self._set_headers()
            self.wfile.write(json.dumps({
                "status": "success",
                "message": "Server is running",
                "kernel_running": kernel_app is not None
            }).encode('utf-8'))
        else:
            self._set_headers()
            self.wfile.write(json.dumps({
                "status": "error",
                "error": f"Unknown endpoint: {self.path}"
            }).encode('utf-8'))

def get_ip():
    """Get the container's IP address."""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(f"Container hostname: {hostname}, IP: {ip}")
        return ip
    except Exception as e:
        print(f"Error getting IP: {e}")
        # Fallback to localhost
        return "127.0.0.1"

def start_kernel_thread(connection_dir="/tmp/kernel"):
    """Start an IPython kernel in a separate thread."""
    global kernel_app, kernel_thread
    
    def run_kernel():
        global kernel_app
        
        if not KERNEL_AVAILABLE:
            print("IPython kernel not available. Skipping kernel start.")
            return
        
        print("Starting IPython kernel...")
        
        # Create a connection file in the shared volume
        connection_file = os.path.join(connection_dir, "kernel-connection.json")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(connection_file), exist_ok=True)
        print(f"Connection file will be at: {connection_file}")
        
        # Launch the kernel
        kernel_app = IPKernelApp.instance()
        kernel_app.connection_file = connection_file
        kernel_app.ip = get_ip()
        
        # Start the kernel
        kernel_app.initialize([])
        kernel_app.start()
    
    # Create and start the thread
    kernel_thread = threading.Thread(target=run_kernel)
    kernel_thread.daemon = True
    kernel_thread.start()
    
    # Give the kernel a moment to start
    time.sleep(2)
    
    return kernel_thread

def is_running_in_docker():
    """Check if the script is running inside a Docker container."""
    return os.path.exists('/.dockerenv')

def run_server(port=8888, start_ipython_kernel=False):
    """Run the HTTP server."""
    # Create a data directory if running in Docker
    if is_running_in_docker():
        try:
            os.makedirs('/app/data', exist_ok=True)
            print("Created /app/data directory for file storage")
        except Exception as e:
            print(f"Warning: Could not create /app/data directory: {e}")
    else:
        # Create a local data directory for testing
        os.makedirs('data', exist_ok=True)
        print("Created local data directory for testing")
    
    # Start IPython kernel if requested
    if start_ipython_kernel:
        start_kernel_thread()
    
    # Start HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, CodeHandler)
    print(f"Starting HTTP server on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
        httpd.server_close()
        
        # Shutdown kernel if it was started
        if kernel_app:
            print("Shutting down kernel...")
            kernel_app.kernel.shutdown()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a Python code execution server')
    parser.add_argument('--port', type=int, default=8888, help='Port to run the server on')
    parser.add_argument('--kernel', action='store_true', help='Start an IPython kernel')
    args = parser.parse_args()
    
    # Set up signal handling for graceful shutdown
    def handle_sigterm(signum, frame):
        print("Received SIGTERM. Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Get the hostname/IP
    get_ip()
    
    # Run the server
    run_server(port=args.port, start_ipython_kernel=args.kernel) 