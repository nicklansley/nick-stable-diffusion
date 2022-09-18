import os
import sys
import json
import copy
import time
from json import JSONDecodeError

import redis
import signal
import requests

METADATA_START = b'##STARTMETADATA##'
METADATA_END = b'##ENDMETADATA##'
ADD_METADATA_TO_FILES = True


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


def send_prompt_request_to_sd_engine(prompt_info):
    try:
        prompt_json = json.dumps(prompt_info)
        print('\nSCHEDULER: Sending prompt request to SD Engine:', prompt_json)
        r = requests.post('http://sd-backend:8080/prompt', json=prompt_json)
        response = r.json()
        print('\nSCHEDULER: send_prompt_request_to_sd_engine - Response from SD Engine:', response)
        return response
    except Exception as e:
        print("SCHEDULER: send_prompt_request_to_sd_engine - Error:", e)
        return {'queue_id': 'X', 'success': False}

def send_upscale_request_to_sd_engine(data):
    try:
        upscale_json = {
            "image_list": data['image_list'],
            "upscale_factor": data['upscale_factor'],
            "queue_id": data['queue_id']
        }
        print('\nSCHEDULER: Sending upscale request to SD Engine:', upscale_json)
        r = requests.post('http://sd-backend:8080/upscale', json=upscale_json)
        response = r.json()
        print('\nSCHEDULER: send_upscale_request_to_sd_engine - Response from SD Engine:', response)
        return response
    except Exception as e:
        print("SCHEDULER: send_upscale_request_to_sd_engine - Error:", e)
        return {'queue_id': 'X', 'success': False}

def update_library_catalogue(queue_id):
    print('\nSCHEDULER: Updating library catalogue')
    library_entry = {}

    # load the current library catalogue
    try:
        with open("/app/library/library.json", "r", encoding="utf8") as library_json_file:
            library = json.load(library_json_file)
    except FileNotFoundError:
        rebuild_library_catalogue()
        return
    except JSONDecodeError as jde:
        rebuild_library_catalogue()
        return

    try:
        # read this queue_id's index card:
        idx_file_name = "/app/library/" + queue_id + "/index.json"
        with open(idx_file_name, "r", encoding="utf8") as index_json_file:
            metadata = json.load(index_json_file)

            # copy the metadata to the library entry
            if type(metadata) is dict:
                library_entry = copy.deepcopy(metadata)
                library_entry["creation_unixtime"] = os.path.getmtime(idx_file_name)
                library_entry["generated_images"] = []

        for root, dirs, files in os.walk("/app/library/" + queue_id, topdown=False):
            add_image_list_entries_to_library_entry(files, library_entry, root)

        # check if the queue_id is already in the library catalogue and replace the entry if it is
        new_library = []
        replaced = False
        for existing_library_entry in library:
            if existing_library_entry["queue_id"] == queue_id:
                print("SCHEDULER: Replacing existing library entry for queue_id:", queue_id)
                new_library.append(json.loads(json.dumps(library_entry)))
                replaced = True
            else:
                print("SCHEDULER: Keeping existing library entry for queue_id:", existing_library_entry["queue_id"])
                new_library.append(existing_library_entry)

        if not replaced:
            print("SCHEDULER: Adding new library entry for queue_id:", queue_id)
            new_library.append(json.loads(json.dumps(library_entry)))

        # write the library catalogue
        with open("/app/library/library.json", "w", encoding="utf8") as outfile:
            outfile.write(json.dumps(new_library, indent=4, sort_keys=True))

        print('\nSCHEDULER: Update of library catalogue completed')

    except json.decoder.JSONDecodeError as jde:
        print("SCHEDULER: update_library_catalogue JSONDecodeError:", jde)

    except Exception as e:
        print('\nSCHEDULER: Update of library catalogue failed', e)


def rebuild_library_catalogue():
    print('\nSCHEDULER: Rebuilding library catalogue')
    library = []
    library_entry = {}

    try:
        # read the library catalogue
        for root, dirs, files in os.walk("/app/library", topdown=False):
            # for each file in the library catalogue
            for idx_name in files:
                # get the file name
                if idx_name == 'index.json':
                    idx_file_name = os.path.join(root, idx_name)
                    unix_time = os.path.getmtime(idx_file_name)
                    try:
                        # read the file
                        with open(idx_file_name, "r", encoding="utf8") as infile:
                            metadata = json.loads(infile.read())

                            # copy the metadata to the library entry
                            if type(metadata) is dict:
                                library_entry = copy.deepcopy(metadata)
                                library_entry["creation_unixtime"] = unix_time
                                library_entry["generated_images"] = []

                                # add the library entry to the library catalogue
                                library.append(json.loads(json.dumps(library_entry)))

                    except json.decoder.JSONDecodeError as jde:
                        print("SCHEDULER: rebuild_library_catalogue JSONDecodeError:", jde)

        # add the images file paths to the library catalogue
        for root, dirs, files in os.walk("/app/library", topdown=False):
            for image_name in files:
                image_path = os.path.join(root, image_name)
                if 'drag_and_drop_images' not in image_path and (image_name.endswith('.jpeg') or image_name.endswith('.jpg') or image_name.endswith('.png')):

                    # if the image has a metadata section then add it to the
                    # end of the image file if ADD_METADATA_TO_FILES is enabled
                    if ADD_METADATA_TO_FILES:
                        update_file_metadata(image_path, library_entry)

                    # add the image file path to the library entry
                    for library_entry in library:
                        if library_entry["queue_id"] in root:
                            image_file_path = image_path.replace('/app/', '')

                            if image_file_path not in library_entry["generated_images"]:
                                library_entry["generated_images"].append(image_file_path)

        # write the library catalogue
        with open("/app/library/library.json", "w", encoding="utf8") as outfile:
            outfile.write(json.dumps(library, indent=4, sort_keys=True))

        print('\nSCHEDULER: Rebuild of library catalogue completed')

    except Exception as e:
        print('\nSCHEDULER: Rebuild of library catalogue failed, or there is no library until first images are created', e)


def add_image_list_entries_to_library_entry(files, library_entry, root):
    # Used by update_library_catalogue() and rebuild_library_catalogue() to add the generated images to library entry
    for image_name in files:
        if image_name.endswith('.jpeg') or image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_file_path = os.path.join(root, image_name)

            # if the image has a metadata section then add it to the
            # end of the image file if ADD_METADATA_TO_FILES is enabled
            if ADD_METADATA_TO_FILES:
                update_file_metadata(image_file_path, library_entry)

            # add the image file path to the library entry
            if library_entry["queue_id"] in root:
                image_file_path = image_file_path.replace('/app/', '')

                if image_file_path not in library_entry["generated_images"]:
                    library_entry["generated_images"].append(image_file_path)


def update_file_metadata(img_path, library_entry):
    # if library_entry is an empty object, it means that an image file found in the library folder is
    # not to do with the library - for example images placed there by the user for
    # local processing of input images in the advanced.html page. In which case just return False
    if library_entry == {}:
        return False

    # make a copy of the library entry so we don't modify the original
    library_metadata = copy.deepcopy(library_entry)

    # remove the generated_images list from the library metadata
    del library_metadata['generated_images']

    # prepare the metadata to be added to the image file
    text = json.dumps(library_metadata)
    binary_text = METADATA_START + text.encode('utf-8') + METADATA_END

    # read the image file ready to insert the metadata
    with open(img_path, 'rb') as image_file:
        image_binary = image_file.read()

    # see if an existing metadata section exists
    prompt_location_start = image_binary.find(METADATA_START)

    if prompt_location_start > -1:
        # replace an existing metadata entry with new metadata entry
        image_binary = image_binary[:prompt_location_start] + binary_text
    else:
        # add new metadata entry to end of image file
        image_binary = image_binary + binary_text

    # write the image file with the new metadata
    with open(img_path, 'wb') as new_image_file:
        new_image_file.write(image_binary)

    # confirm the metadata was added to the image file
    return read_file_prompt(img_path) == text


def read_file_prompt(img_path):
    # read the image file ready to extract the metadata
    with open(img_path, 'rb') as image_file:
        image_binary = image_file.read()

    # see if an existing metadata section exists
    prompt_location_start = image_binary.find(METADATA_START)
    if prompt_location_start > 0:
        prompt_location_end = image_binary.find(METADATA_END)
        if prompt_location_end > 0:
            # extract the metadata from the image file
            prompt_section = image_binary[prompt_location_start + len(METADATA_START):prompt_location_end]
            return prompt_section.decode('utf-8')

    # no metadata found
    return ''


def exit_signal_handler(self, sig):
    sys.stderr.write('\nSCHEDULER: Shutting down...\n')
    sys.stderr.flush()
    quit()


if __name__ == "__main__":
    print("SCHEDULER: Started")
    signal.signal(signal.SIGTERM, exit_signal_handler)
    signal.signal(signal.SIGINT, exit_signal_handler)
    rebuild_library_catalogue()
    print("SCHEDULER: Listening for requests...")
    while True:
        time.sleep(1)
        queue_item = get_next_queue_request()
        if queue_item['queue_id'] != 'X':
            if queue_item['command'] == 'prompt':
                request_data = send_prompt_request_to_sd_engine(queue_item)
            elif queue_item['command'] == 'upscale':
                request_data = send_upscale_request_to_sd_engine(queue_item)

            delete_request_from_redis_queue(queue_item)

            if request_data['queue_id'] == queue_item['queue_id']:
                update_library_catalogue(queue_item['queue_id'])

            if request_data['success']:
                print("SCHEDULER: Processing complete - library updated\n\n")
            else:
                print("SCHEDULER: Request failed")
