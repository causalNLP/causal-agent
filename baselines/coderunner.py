import os
import time
import docker
import shlex
import json
import socket
import requests
import base64
from jupyter_client import BlockingKernelClient
from queue import Empty


class CodeRunner:
    def __init__(self, safe_exec=True, persistent=False, session_timeout=3600):
        self.client = docker.from_env()
        self.safe_exec = safe_exec
        self.persistent = persistent
        self.session_timeout = session_timeout
        self.name = "python-causalscientistrunner"
        self.http_name = "python-causalscientist-http"
        self.kernel_client = None
        self.session_start_time = None
        self.connection_file_host = "/tmp/causalscientist/kernel-connection.json"
        self.http_port = 8888
        self.http_container = None
        self.http_url = None
        
        # Create directory for connection file if it doesn't exist
        os.makedirs(os.path.dirname(self.connection_file_host), exist_ok=True)

    def delete_container(self, container_name=None):
        """Delete a container by name."""
        name = container_name or self.name
        try:
            container = self.client.containers.get(name)
            if container is not None:
                container.remove(force=True)
        except:
            pass
            
    def start_persistent_container(self):
        """Start a persistent container with HTTP server."""
        # Delete any existing container
        self.delete_container(self.http_name)
        
        # Create a new container
        try:
            print(f"Creating container with image: {self.http_name}")
            
            container = self.client.containers.run(
                self.http_name,
                detach=True,
                name=self.http_name,
                ports={'8888/tcp': self.http_port},
                remove=False
            )
            
            self.http_container = container
            print(f"Container created with ID: {container.id}")
            
            # Wait for the HTTP server to start
            max_wait = 30  # seconds
            start_time = time.time()
            self.http_url = f"http://localhost:{self.http_port}"
            
            print(f"Waiting for HTTP server to start at {self.http_url}...")
            
            while time.time() - start_time < max_wait:
                try:
                    # Check container logs
                    if (time.time() - start_time) % 5 == 0:  # Every 5 seconds
                        logs = container.logs().decode('utf-8')
                        print(f"Container logs:\n{logs}")
                    
                    # Try to connect to the health endpoint
                    response = requests.get(f"{self.http_url}/health", timeout=1)
                    if response.status_code == 200:
                        print("HTTP server is running!")
                        self.session_start_time = time.time()
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
                    print(f"Waiting for HTTP server... ({int(time.time() - start_time)}s)")
            
            print("Timed out waiting for HTTP server to start")
            self.delete_container(self.http_name)
            return False
            
        except Exception as e:
            print(f"Error starting persistent container: {str(e)}")
            import traceback
            traceback.print_exc()
            self.delete_container(self.http_name)
            return False
            
    def stop_persistent_container(self):
        """Stop the persistent container."""
        self.delete_container(self.http_name)
        self.http_container = None
        self.http_url = None
        self.session_start_time = None
                
    def is_container_running(self):
        """Check if the persistent container is running."""
        if not self.http_container:
            return False
            
        try:
            self.http_container.reload()
            return self.http_container.status == "running"
        except:
            return False
            
    def check_session_timeout(self):
        """Check if the session has timed out."""
        if not self.session_start_time:
            return False
            
        return (time.time() - self.session_start_time) > self.session_timeout
        
    def run_code(self, code, help=False, persistent=None):
        """Runs code in a docker container, outputs the result.
        For safe execution, use a container.

        Args:
        code: str - the code to run
        help: bool - if True, run as shell command (for one-off mode only)
        persistent: bool - override instance setting for persistent mode
        """
        use_persistent = self.persistent if persistent is None else persistent
        
        if use_persistent:
            return self.run_code_persistent(code)
        else:
            return self.run_code_oneoff(code, help)
            
    def run_code_persistent(self, code):
        """Run code in the persistent container using HTTP server."""
        # Check if container is running, start if not
        if not self.is_container_running():
            if not self.start_persistent_container():
                return "Error: Failed to start persistent container"
                
        # Check for session timeout
        if self.check_session_timeout():
            print("Session timed out, restarting container...")
            self.stop_persistent_container()
            if not self.start_persistent_container():
                return "Error: Session timed out and failed to restart"
                
        # Execute code
        try:
            response = requests.post(
                f"{self.http_url}/execute",
                json={"code": code},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    output = result["output"]
                    error = result["error"]
                    if error:
                        return f"{output}\n{error}"
                    else:
                        return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return f"Error: HTTP status {response.status_code}"
                
        except Exception as e:
            return f"Error executing code: {str(e)}"

    def run_code_oneoff(self, code, help=False):
        """Runs code in a one-off container (existing implementation)."""
        if self.safe_exec:
            # create dir if it doesn't exist
            if not os.path.exists("/tmp/causalscientistrunner"):
                os.makedirs("/tmp/causalscientistrunner")
            # create a tmp file with the code
            exec_file_path = "/tmp/causalscientistrunner/tmp_{}.py".format(
                time.strftime("%Y%m%d%H%M%S")
            )
            with open(exec_file_path, "w") as f:
                f.write(code)

            command = f"timeout 180s python {exec_file_path}"

            if help:
                # Only run code in bash
                command = code

            try:
                self.delete_container()
                logs = self.client.containers.run(
                    "python-causalscientistrunner",
                    detach=False,
                    name="python-causalscientistrunner",
                    command=command,
                    stdout=True,
                    stderr=True,
                    remove=True,
                    volumes={
                        exec_file_path: {
                            "bind": exec_file_path,
                            "mode": "ro",  # Read-only
                        }
                    },
                )
            except Exception as e:
                return str(e)

            # remove tmp file if it exists
            if os.path.exists(exec_file_path):
                os.remove(exec_file_path)

            return logs.decode("utf-8")

        else:
            print("Warning: Running code without a container.")
            try:
                res = exec(code)
            except Exception as e:
                return str(e)
            return res
            
    def get_variable_value(self, variable_name):
        """Get the value of a variable in the persistent environment."""
        if not self.is_container_running():
            return "Error: No active persistent session"
            
        try:
            response = requests.post(
                f"{self.http_url}/variable",
                json={"name": variable_name},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    return result["value"]
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return f"Error: HTTP status {response.status_code}"
                
        except Exception as e:
            return f"Error getting variable: {str(e)}"
        
    def get_defined_variables(self):
        """Get a list of all defined variables in the environment."""
        if not self.is_container_running():
            return "Error: No active persistent session"
            
        try:
            response = requests.post(
                f"{self.http_url}/variables",
                json={},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    return str(result["variables"])
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return f"Error: HTTP status {response.status_code}"
                
        except Exception as e:
            return f"Error getting variables: {str(e)}"
            
    def get_variables(self):
        """
        Get a dictionary of all defined variables in the environment.
        
        Returns:
            A dictionary of variable names or an empty dict if not in persistent mode
        """
        if not self.persistent or not self.is_container_running():
            return {}
            
        try:
            variables_str = self.get_defined_variables()
            
            # Check if the result is an error message
            if isinstance(variables_str, str) and variables_str.startswith("Error:"):
                print(f"Warning: {variables_str}")
                return {}
            
            # Parse the string representation of the dictionary
            if isinstance(variables_str, str):
                import ast
                variables_dict = ast.literal_eval(variables_str)
                return {var: True for var in variables_dict}
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting variables: {str(e)}")
            return {}
            
    def upload_file(self, local_path, container_path=None):
        """Upload a file from the local machine to the container."""
        if not self.is_container_running():
            return "Error: No active persistent session"
            
        try:
            # If container_path is not specified, use the same path as local_path
            if container_path is None:
                container_path = os.path.basename(local_path)
                
            # Read the file content
            with open(local_path, 'rb') as f:
                file_content = f.read()
                
            # Encode file content as base64
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            
            # Send the file to the container
            response = requests.post(
                f"{self.http_url}/upload_file",
                json={
                    "filename": container_path,
                    "content": encoded_content
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    return f"File uploaded successfully to {container_path}"
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return f"Error: HTTP status {response.status_code}"
                
        except Exception as e:
            return f"Error uploading file: {str(e)}"
            
    def download_file(self, container_path, local_path=None):
        """Download a file from the container to the local machine."""
        if not self.is_container_running():
            return "Error: No active persistent session"
            
        try:
            # If local_path is not specified, use the same path as container_path
            if local_path is None:
                local_path = os.path.basename(container_path)
                
            # Request the file from the container
            response = requests.post(
                f"{self.http_url}/download_file",
                json={"filename": container_path},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    # Decode the file content
                    file_content = base64.b64decode(result["content"])
                    
                    # Create directories if they don't exist
                    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
                    
                    # Write the file
                    with open(local_path, 'wb') as f:
                        f.write(file_content)
                        
                    return f"File downloaded successfully to {local_path}"
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return f"Error: HTTP status {response.status_code}"
                
        except Exception as e:
            return f"Error downloading file: {str(e)}"
            
    def list_files(self, directory='.'):
        """List files in a directory in the container."""
        if not self.is_container_running():
            return "Error: No active persistent session"
            
        try:
            response = requests.post(
                f"{self.http_url}/list_files",
                json={"directory": directory},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    return result["files"]
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return f"Error: HTTP status {response.status_code}"
                
        except Exception as e:
            return f"Error listing files: {str(e)}"


if __name__ == "__main__":
    code = """
import pandas as pd
import dowhy
import numpy as np
from scipy import stats

# Hypothetical data (replace with your actual employment data)
data = {'state': np.random.choice([0, 1], size=410),
        'fte': np.random.rand(410) * 50,
        'fte_after': np.random.rand(410) * 50,
        'employment': np.random.randint(100, 500, size=410),
        'employment_after': np.random.randint(100, 500, size=410)}
df = pd.DataFrame(data)



#Difference-in-Differences (DID) estimation (assuming we had employment data)
df['employment_change'] = df['employment_after'] - df['employment']
treatment_effect = df.groupby('state')['employment_change'].mean().diff().iloc[1]


print(f"Difference-in-Differences estimate of employment change: {treatment_effect}")
"""
    runner = CodeRunner()
    res = runner.run_code(code)
    print(res)
