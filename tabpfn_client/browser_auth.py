#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
import threading
from threading import Event
import http.server
import socketserver
import webbrowser
import urllib.parse
from typing import Optional, Tuple
import time
from urllib.parse import quote
import ssl
from pathlib import Path
import os

BASE_PATH = Path(__file__).parent.resolve()
PROJECT_ROOT = os.path.dirname(BASE_PATH)
class BrowserAuthHandler:
    def __init__(self, gui_url: str):
        self.gui_url = gui_url
        self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        print(PROJECT_ROOT)
        self.ssl_context.load_cert_chain(certfile= os.path.join(BASE_PATH, "certs", "localhost.pem"), keyfile= os.path.join(BASE_PATH, "certs", "localhost-key.pem"))

    def try_browser_login(self) -> Tuple[bool, Optional[str]]:
        """
        Attempts to perform browser-based login
        Returns (success: bool, token: Optional[str])
        """
        auth_event = Event()
        received_token = None
        timeout_event = Event()

        class CallbackHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                nonlocal received_token

                #  Check if this is a redirect from HTTPS
                if self.path.startswith("/login") or self.path == "/":
                    # This might be Safari trying to access the root after a redirect
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    redirect_html = f"""
                    <html><head>
                    <script>
                    // Try to extract token from URL or parent window
                    const urlParams = new URLSearchParams(window.location.search);
                    const token = urlParams.get('token');
                    if (token) {{
                        window.location = "/?token=" + token;
                    }}
                    </script>
                    </head><body>Redirecting...</body></html>
                    """
                    self.wfile.write(redirect_html.encode())
                    return

                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)

                if "token" in query:
                    received_token = query["token"][0]

                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "*"))
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Vary", "Origin") # Allow CORS preflight
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.end_headers()
                success_html = """
                <html>
                    <body style="text-align: center; font-family: Arial, sans-serif; padding: 50px;">
                        <h2>Login successful!</h2>
                        <p>You can close this window and return to your application.</p>
                    </body>
                </html>
                """
                self.wfile.write(success_html.encode())
                auth_event.set()
            
            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.end_headers()

            def log_message(self, format, *args):
                pass

        try:
            with socketserver.TCPServer(("127.0.0.1", 0), CallbackHandler) as httpd:
                httpd.socket = self.ssl_context.wrap_socket(httpd.socket, server_side=True, do_handshake_on_connect=True)
                port = httpd.server_address[1]
                callback_url = f"https://127.0.0.1:{port}"
                
                login_url = f"{self.gui_url}/login?callback={urllib.parse.quote(callback_url, safe=':/')}"
                # login_url = f"{self.gui_url}/login?callback={callback_url}"

                print(
                    "\nOpening browser for login. Please complete the login/registration process in your browser and return here.\n"
                )

                if not webbrowser.open(login_url):
                    print(
                        "\nCould not open browser automatically. Falling back to command-line login...\n"
                    )
                    return False, None

                # Start timeout handler after server is running
                def timeout_handler():
                    time.sleep(300)  # 5 minutes timeout
                    if not auth_event.is_set():
                        timeout_event.set()
                        httpd.server_close()

                timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
                timeout_thread.start()

                while not auth_event.is_set() and not timeout_event.is_set():
                    httpd.handle_request()

                return received_token is not None, received_token

        except Exception as e:
            print(f"\nBrowser auth failed: {str(e)}. Falling back to command-line login...\n")
            return False, None
        except Exception:
            print("\n Browser auth failed. Falling back to command-line login...\n")
            return False, None
