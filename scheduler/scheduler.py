import os
import sys
import json
import time
import redis
import signal
import requests


def get_next_queue_request():
    try:
        r = redis.Redis(host='scheduler', port=6379, db=0, password='hellothere')
        queue_list = []
        queue_data = r.lrange('queue', 0, -1)
        if len(queue_data) > 0:
            for current_queue_item in queue_data:
                queue_list.append(json.loads(current_queue_item.decode()))
            queue_list.reverse()
            return queue_list[0]
        else:
            # Return an 'empty' queue item which will be used to stop the scheduler trying to delete a non-existent queue item
            return {'queue_id': 'X'}

    except Exception as e:
        print("SCHEDULER: get_next_queue_request Error:", e)
        return {'queue_id': 'X'}


def delete_request_from_redis_queue(queue_data):
    try:
        r = redis.Redis(host='scheduler', port=6379, db=0, password='hellothere')
        print('\nSCHEDULER: Deleting queue item:', queue_data)
        r.lrem('queue', 0, json.dumps(queue_data))
        return True
    except redis.exceptions.ConnectionError as ce:
        print("SCHEDULER: delete_request_from_redis_queue Connection Error:", ce)
        return False
    except Exception as e:
        print("SCHEDULER: delete_request_from_redis_queue Error:", e)
        return False


def send_request_to_sd_engine(prompt_info):
    try:
        prompt_json = json.dumps(prompt_info)
        print('\nSCHEDULER: Sending json request to SD Engine:', prompt_json)
        r = requests.post('http://sd-backend:8080/prompt', json=prompt_json)
        response = r.json()
        print('\nSCHEDULER: send_request_to_sd_engine - Response from SD Engine:', response)
        return response
    except Exception as e:
        print("SCHEDULER: send_request_to_sd_engine - Error:", e)
        return {'queue_id': 'X', 'success': False}


def update_library_catalogue():
    print('\nSCHEDULER: Updating library catalogue')
    library = []
    library_entry = {
        "text_prompt": "",
        "queue_id": "",
        "generated_images": []
    }
    for root, dirs, files in os.walk("/app/library", topdown=False):
        for idx_name in files:
            if idx_name == 'index.json':
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


def exit_signal_handler(self, sig):
    sys.stderr.write('\nSCHEDULER: Shutting down...\n')
    sys.stderr.flush()
    quit()


if __name__ == "__main__":
    print("SCHEDULER: Started")
    signal.signal(signal.SIGTERM, exit_signal_handler)
    signal.signal(signal.SIGINT, exit_signal_handler)
    update_library_catalogue()
    print("SCHEDULER: Listening for requests...")
    while True:
        time.sleep(1)
        queue_item = get_next_queue_request()
        if queue_item['queue_id'] != 'X':
            request_data = send_request_to_sd_engine(queue_item)
            if request_data['queue_id'] == queue_item['queue_id']:
                update_library_catalogue()

            if request_data['success']:
                print("SCHEDULER: Processing complete - library updated\n\n")
            else:
                print("SCHEDULER: Request failed")

            delete_request_from_redis_queue(queue_item)
