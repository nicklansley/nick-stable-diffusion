import json
import os
import random
from urllib.parse import unquote
import signal
import sys
import redis
import uuid

from http.server import BaseHTTPRequestHandler, HTTPServer


def update_library_catalogue():
    print('\nFROMTEND: Updating library catalogue')
    library = []
    library_entry = {
        "text_prompt": "",
        "queue_id": "",
        "generated_images": []
    }
    for root, dirs, files in os.walk("/app/library", topdown=False):
        for idx_name in files:
            if idx_name.endswith('.json'):
                idx_file_name = os.path.join(root, idx_name)
                unix_time = os.path.getmtime(idx_file_name)
                try:
                    with open(idx_file_name, "r", encoding="utf8") as infile:
                        metadata = json.loads(infile.read())
                        if type(metadata) is dict:
                            library_entry["text_prompt"] = metadata["text_prompt"]
                            library_entry["queue_id"] = metadata["queue_id"]
                            library_entry["seed"] = metadata["seed"]
                            library_entry["creation_unixtime"] = unix_time
                            library_entry["process_time_secs"] = metadata["time_taken"]
                            library_entry["generated_images"] = []
                            library_entry['height'] = metadata['height']
                            library_entry['width'] = metadata['width']
                            library_entry['ddim_steps'] = metadata['ddim_steps']
                            library_entry['ddim_eta'] = metadata['ddim_eta']
                            library_entry['scale'] = metadata['scale']
                            library_entry['downsampling_factor'] = metadata['downsampling_factor']
                            library_entry['error'] = metadata['error']

                            library.append(json.loads(json.dumps(library_entry)))
                except json.decoder.JSONDecodeError as jde:
                    print("SCHEDULER: update_library_catalogue JSONDecodeError:", jde)

    for root, dirs, files in os.walk("/app/library", topdown=False):
        for image_name in files:
            if image_name.endswith('.jpeg') or image_name.endswith('.jpg') or image_name.endswith('.png'):
                for library_entry in library:
                    if library_entry["queue_id"] in root:
                        image_file_path = os.path.join(root, image_name).replace('/app/', '')
                        if image_file_path not in library_entry["generated_images"]:
                            library_entry["generated_images"].append(image_file_path)

    for library_entry in library:
        if len(library_entry["generated_images"]) == 0:
            library.remove(library_entry)

    with open("/app/library/library.json", "w", encoding="utf8") as outfile:
        outfile.write(json.dumps(library, indent=4, sort_keys=True))

    print('\nSCHEDULER: Update of library catalogue completed')


class RelayServer(BaseHTTPRequestHandler):
    def do_GET(self):
        api_command = unquote(self.path)
        print("\nFRONTEND: GET >> API command =", api_command)
        if api_command.endswith('/'):
            self.process_ui('/index.html')
        elif api_command.endswith('/getlibrary'):
            self.process_ui('/library/library.json')
        elif api_command.endswith('.html') or \
                api_command.endswith('.js') or \
                api_command.endswith('.css') or \
                api_command.endswith('.ico') or \
                api_command.endswith('.gif') or \
                api_command.endswith('.png') or \
                api_command.endswith('.jpeg') or \
                api_command.endswith('.jpg'):
            self.process_ui(api_command)
        elif api_command == '/queue_status':
            queue_data = self.check_queue_request()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response_body = json.dumps(queue_data)
            self.wfile.write(response_body.encode())
        return

    def do_POST(self):
        api_command = unquote(self.path)
        print("\nFRONTEND: POST >> API command =", api_command)
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)
        print("\nFRONTEND:", data)
        if api_command == '/prompt':
            result = self.queue_request_to_redis(data)
            if result == 'X':
                self.send_response(500)
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response_body = '{"queue_id": "' + result + '"}'
                self.wfile.write(response_body.encode())

        elif api_command == '/deleteimage':
            if self.process_deleteimage(data):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"success": true}')
            else:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"success": false}')
        return

    def queue_request_to_redis(self, data):
        try:
            r = redis.Redis(host='scheduler', port=6379, db=0, password='hellothere')
            data['queue_id'] = str(uuid.uuid4())
            data['num_images'] = int(data['num_images'])
            data['seed'] = int(data['seed'])

            # These are optional parameters
            try:
                if not data['gen_top_k']:
                    pass
                else:
                    data['gen_top_k'] = float(data['gen_top_k'])
            except KeyError:
                pass

            try:
                if not data['gen_top_p']:
                    pass
                else:
                    data['gen_top_p'] = float(data['gen_top_p'])
            except KeyError:
                pass

            try:
                if not data['temperature']:
                    pass
                else:
                    data['temperature'] = float(data['temperature'])
                if data['temperature'] == 0.0:
                    data['temperature'] = 0.01
            except KeyError:
                pass

            try:
                if not data['condition_scale']:
                    pass
                else:
                    data['condition_scale'] = float(data['condition_scale'])
            except KeyError:
                pass

            r.lpush('queue', json.dumps(data))
            print("\nFRONTEND: Request queued to redis with queue_id:", data['queue_id'])
            return data['queue_id']
        except Exception as e:
            print("\nFRONTEND: queue_request_to_redis Error:", e)
            return 'X'

    def check_queue_request(self):
        try:
            r = redis.Redis(host='scheduler', port=6379, db=0, password='hellothere')
            queue_list = []
            queue_data = r.lrange('queue', 0, -1)
            for queue_item in queue_data:
                queue_list.append(json.loads(queue_item.decode()))
            queue_list.reverse()
            return queue_list
        except Exception as e:
            print("\nFRONTEND: check_queue_request Error:", e)
            return []

    def process_deleteimage(self, data):
        try:
            os.remove('/app/' + data['path'])
            print('\nFRONTEND: /app/' + data['path'] + " deleted")
            update_library_catalogue()
        except FileNotFoundError as fnfe:
            print("\nFRONTEND: process_deleteimage FileNotFoundError:", fnfe)
            return False
        return True

    def process_ui(self, path):
        if path.endswith('.html'):
            response_content_type = 'text/html'
        elif path.endswith('.js'):
            response_content_type = 'text/javascript'
        elif path.endswith('.css'):
            response_content_type = 'text/css'
        elif path.endswith('.ico'):
            response_content_type = 'image/x-icon'
        elif path.endswith('.gif'):
            response_content_type = 'image/gif'
        elif path.endswith('.png'):
            response_content_type = 'image/png'
        elif path.endswith('.jpg'):
            response_content_type = 'image/jpeg'
        else:
            response_content_type = 'text/plain'

        file_path = '/app' + path

        try:
            with open(file_path, 'rb') as data_file:
                data = data_file.read()
                self.log_message(file_path + ' file read successfully')
                self.send_response(200)
                self.send_header('Content-Type', response_content_type)
                self.end_headers()
                self.wfile.write(data)

        except FileNotFoundError:
            self.log_message(file_path + ' file not found')
            self.send_response(404)
            self.end_headers()


def exit_signal_handler(self, sig):
    sys.stderr.write('Shutting down...\n')
    sys.stderr.flush()
    quit()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, exit_signal_handler)
    signal.signal(signal.SIGINT, exit_signal_handler)
    relayServerRef = HTTPServer(("", 3000), RelayServer)
    sys.stderr.write('Frontend Web Server\n\n')
    sys.stderr.flush()

    try:
        relayServerRef.serve_forever()
    except KeyboardInterrupt:
        pass

    relayServerRef.server_close()
    sys.stderr.write('Relay Server Stopped Successfully\n')
