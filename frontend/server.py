import json
import os
import random
from urllib.parse import unquote
import signal
import sys
import redis
import uuid
import base64

from http.server import BaseHTTPRequestHandler, HTTPServer

PORT_NUMBER = 3000

class RelayServer(BaseHTTPRequestHandler):
    def do_GET(self):
        api_command = unquote(self.path)

        # if the URL includes query parameters, remove them!
        # We don't use query strings in this app but the web client sends a timestamp in a query string
        # such as 'wantedimage.png?&timestamp=166264887209' to overcome browser caching

        query_string = api_command.split('?')
        if len(query_string) > 1:
            api_command = api_command.split('?')[0]

        print("\nFRONTEND: GET >> API command =", api_command)
        if api_command.endswith('/getlibrary'):
            self.process_ui('/library/library.json')

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

        if api_command == '/prompt' or api_command == '/upscale':
            result = 'X'
            ok_flag, message, data = self.quality_assure(data)
            if ok_flag:
                if api_command == '/prompt':
                    result = self.queue_prompt_request_to_redis(data)
                else:
                    result = self.queue_upscale_request_to_redis(data)
            else:
                self.send_response(400)
                self.end_headers()

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

        elif api_command == '/imagelist':
            if 'queue_id' in data:
                image_list = self.get_image_list(data['queue_id'])
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(image_list).encode())
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"success": false}')

        return


    # take base64 string and save it to a file as binary
    # the image string starts 'data:image/png;base64,iVBORw....' in base64 encoding
    def convert_base64_to_image_binary_and_save_file(self, image_base64_string):
        try:
            image_elements = image_base64_string.split(',')
            image_suffix = image_elements[0].split(';')[0].split('/')[1]
            confirm_base64 = image_elements[0].split(';')[1] == 'base64'
            if confirm_base64:
                # encode the base64 string to binary
                image_data = base64.b64decode(image_elements[1])
                # think up a random 8 alphanumeric character filename so it doesn't overwrite anything
                image_name = str(uuid.uuid4())[:8] + '.' + image_suffix

                # if a folder does not exist, create it
                if not os.path.exists('/app/library/drag_and_drop_images'):    # if the folder does not exist
                    os.makedirs('/app/library/drag_and_drop_images')

                # save the image file
                image_path = '/app/library/drag_and_drop_images/' + image_name
                print('Image converted to binary and saved to', image_path)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                return image_path.replace('/app/', '')
            else:
                return 'ERROR: Not a base64 string-based imaga'

        except Exception as e:
            print("\nFRONTEND: process_saveimage Error:", e)
            return 'ERROR: Could not process image string'

    def quality_assure(self, data):
        if 'original_image_path' in data and data['original_image_path'] != '' and 'data:image' in data['original_image_path']:
            image_path = self.convert_base64_to_image_binary_and_save_file(data['original_image_path'])
            if image_path.startswith('ERROR'):
                return False, image_path, data
            else:
                data['original_image_path'] = image_path

        return True, 'OK', data

    def queue_prompt_request_to_redis(self, data):
        try:
            r = redis.Redis(host='scheduler', port=6379, db=0, password='hellothere')
            data['command'] = 'prompt'
            data['queue_id'] = str(uuid.uuid4())
            data['num_images'] = int(data['num_images'])
            data['seed'] = int(data['seed'])

            r.lpush('queue', json.dumps(data))
            print("\nFRONTEND: Prompt request queued to redis with queue_id:", data['queue_id'])
            return data['queue_id']
        except Exception as e:
            print("\nFRONTEND: queue_prompt_request_to_redis Error:", e)
            return 'X'

    def queue_upscale_request_to_redis(self, data):
        try:
            r = redis.Redis(host='scheduler', port=6379, db=0, password='hellothere')
            data['command'] = 'upscale'
            data['queue_id'] = data['image_list'][0].split('/')[1].split('.')[0] # gets queue_id from 1st image in list
            data['image_list'] = data['image_list']
            data['upscale_factor'] = int(data['upscale_factor'])

            r.lpush('queue', json.dumps(data))
            print("\nFRONTEND: Upscale request queued to redis with queue_id:", data['queue_id'])
            return data['queue_id']
        except Exception as e:
            print("\nFRONTEND: queue_upscale_request_to_redis Error:", e)
            return 'X'


    def get_image_list(self, queue_id):
        image_data = {
            'completed': False,
            'images': []
        }
        for root, dirs, files in os.walk("/app/library/" + queue_id, topdown=False):
            for image_name in files:
                if image_name.endswith('.png') or image_name.endswith('.jpg'):
                    image_data['images'].append('/library/' + queue_id + '/' + image_name)
                elif image_name == 'index.json':
                    image_data['completed'] = True

        print('\nFRONTEND: Image list for queue_id', queue_id, 'is', image_data)
        return image_data

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
            os.remove(data['path'])
            print('\nFRONTEND:', data['path'], ' deleted')

        except FileNotFoundError as fnfe:
            print("\nFRONTEND: process_deleteimage FileNotFoundError:", fnfe)
            return False
        return True

    def process_ui(self, path):
        print('\nFRONTEND: Serving UI file:', path)
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
        elif path.endswith('.map'):
            response_content_type = 'application/json'
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
            print(file_path + ' file not found')
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
    relayServerRef = HTTPServer(("", PORT_NUMBER), RelayServer)
    sys.stderr.write('Frontend Web Server\n\n')
    sys.stderr.flush()

    try:
        relayServerRef.serve_forever()
    except KeyboardInterrupt:
        pass

    relayServerRef.server_close()
    sys.stderr.write('Relay Server Stopped Successfully\n')
