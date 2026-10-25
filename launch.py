"""Open the existing workspace, or launch it in this terminal."""
import json
import urllib.request
import webbrowser
import threading

URL = 'http://127.0.0.1:5000'

def main():
    try:
        with urllib.request.urlopen(URL + '/health', timeout=2) as response:
            running = json.load(response)
        if running.get('service') == 'loanlens':
            webbrowser.open(URL)
            print('LoanLens is already running. Opened the website in your browser.')
            return
    except (OSError, ValueError):
        pass
    from app import app
    threading.Timer(2, lambda: webbrowser.open(URL)).start()
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    main()
