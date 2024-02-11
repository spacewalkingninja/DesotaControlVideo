import os, sys
import time, re, json, shutil
import requests, subprocess, random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-mr", "--model_req", 
                    help="DeSOTA Request as yaml file path",
                    type=str)
parser.add_argument("-mru", "--model_res_url",
                    help="DeSOTA API Result URL. Recognize path instead of url for desota tests", # check how is atribuited the dev_mode variable in main function
                    type=str)

from requests.adapters import HTTPAdapter, Retry

s = requests.Session()

retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])

s.mount('https://', HTTPAdapter(max_retries=retries))

DEBUG = False
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
# DeSOTA Funcs [START]
#   > Import DeSOTA Scripts
from desota import detools
#   > Grab DeSOTA Paths
USER_SYS = detools.get_platform()
APP_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(CURRENT_PATH, f"tmp")
#IN_PATH = os.path.join(CURRENT_PATH, f"in")
#   > USER_PATH
if USER_SYS == "win":
    path_split = str(APP_PATH).split("\\")
    desota_idx = [ps.lower() for ps in path_split].index("desota")
    USER=path_split[desota_idx-1]
    USER_PATH = "\\".join(path_split[:desota_idx])
elif USER_SYS == "lin":
    path_split = str(APP_PATH).split("/")
    desota_idx = [ps.lower() for ps in path_split].index("desota")
    USER=path_split[desota_idx-1]
    USER_PATH = "/".join(path_split[:desota_idx])
DESOTA_ROOT_PATH = os.path.join(USER_PATH, "Desota")
CONFIG_PATH = os.path.join(DESOTA_ROOT_PATH, "Configs")
SERV_CONF_PATH = os.path.join(CONFIG_PATH, "services.config.yaml")
# DeSOTA Funcs [END]



def main(args):
    '''
    return codes:
    0 = SUCESS
    1 = INPUT ERROR
    2 = OUTPUT ERROR
    3 = API RESPONSE ERROR
    9 = REINSTALL MODEL (critical fail)
    '''
   # Time when grabed
    _report_start_time = time.time()
    start_time = int(_report_start_time)

    #---INPUT---# TODO (PRO ARGS)
    _resnum = 5
    #---INPUT---#

    # DeSOTA Model Request
    model_request_dict = detools.get_model_req(args.model_req)
    
    # API Response URL
    send_task_url = args.model_res_url
    
    # TARGET File Path
    out_filename = f"shuffled-video-{start_time}.mp4"
    out_filepath = os.path.join(TMP_PATH, out_filename)
    
    out_urls = detools.get_url_from_str(send_task_url)
    if len(out_urls)==0:
        dev_mode = True
        report_path = send_task_url
    else:
        dev_mode = False
        report_path = out_urls[0]

    # Get text from request
    _req_text = detools.get_request_text(model_request_dict)
    if isinstance(_req_text, list):
        _req_text = " OR ".join(_req_text)
    if DEBUG:
        with open(os.path.join(APP_PATH, "debug.txt"), "w") as fw:
            fw.writelines([
                f"INPUT: '{_req_text}'\n",
                f"IsINPUT?: {True if _req_text else False}\n"
            ])
    
    
    # TODO Get VIDEO from request TODO
    ##TODO##
    _req_video = detools.get_request_video(model_request_dict) ##TODO##
    #print(model_request_dict)
    if isinstance(_req_video, list):
        _req_video = str(_req_video[0])
    #REMOVE OLD INPUTS
    #try:
    #    shutil.rmtree(IN_PATH)
    #except OSError as e:
    #    print("Error: %s - %s." % (e.filename, e.strerror))
    #os.makedirs(args.IN_PATH, exist_ok=True)

    filename = os.path.basename(_req_video)
    file_ext = os.path.splitext(filename)[1]

    # INPUT File Path    
    #in_filename = f'video-input.{file_ext}'
    #in_filepath = os.path.join(IN_PATH, in_filename)

    ##TODO##
    #with requests.get(_req_video, stream=True) as r:
    #        with open(in_filepath, 'wb') as f:
    #            shutil.copyfileobj(r.raw, f)
    

    # Run Model
    if _req_text:
        _model_run = os.path.join(APP_PATH, "main.py")
        if USER_SYS == "win":
            _model_runner_py = os.path.join(APP_PATH, "env", "python.exe")
        elif USER_SYS == "lin":
            _model_runner_py = os.path.join(APP_PATH, "env", "bin", "python3")


        targs = {}
        if 'prompt' in targs:
            if targs['prompt'] == '-=#{([$argument$])}#=-':
                targs['prompt'] = _req_text
        else:
            targs['prompt'] = _req_text

        if 'condition' not in targs:
            targs['condition'] = "shuffle"
        else:
            if targs['condition'] == "pose":
                targs['condition'] = "openpose"
            if targs['condition'] == "scribble":
                targs['condition'] = "scribble_hedsafe"
            if targs['condition'] == "softedge":
                targs['condition'] = "softedge_hedsafe"
            if targs['condition'] == "face-geometry":
                targs['condition'] = "mediapipe_face"
            if targs['condition'] == "normals":
                targs['condition'] = "normal_bae"
            if targs['condition'] == "geometry":
                targs['condition'] = "mlsd"
            if targs['condition'] == "lineart":
                targs['condition'] = "lineart_realistic"
            if targs['condition'] == "anime":
                targs['condition'] = "lineart_anime"
            if targs['condition'] == "canny":
                targs['condition'] = "canny"
            if targs['condition'] == "shuffle":
                targs['condition'] = "shuffle"
            if targs['condition'] == "depth":
                targs['condition'] = "depth_midas"

        if 'width' not in targs:
            targs['width'] = "256"
        if 'height' not in targs:
            targs['height'] = "256"
        if 'video_length' not in targs:
            targs['video_length'] = "15"
        if 'frame_rate' not in targs:
            targs['frame_rate'] = "2"
        if 'seed' in targs:
            targs['seed'] = str(targs.seed) if targs.seed.isdigit() else str(random.randint(1, 1000000))
        if 'seed' not in targs:
            targs['seed'] = str(random.randint(1, 1000000))

        targs['version'] = "v11"
        print(targs)
        le_cmd = [
            _model_runner_py, _model_run, 
            "--prompt", f'"{targs["prompt"]}"', 
            "--resnum", str(_resnum),
            "--respath", str(out_filename),
            "--video_path", str(_req_video),
            "--condition", str(targs["condition"]),
            "--video_length", str(targs["video_length"]),
            "--width", str(targs['width']),
            "--height", str(targs['height']),
            "--frame_rate", str(targs["frame_rate"]),
            "--version",  str(targs["version"]),  # You may need to adjust this
            "--seed", str(targs["seed"]) ,
            #"--is_long_video" if targs.is_long_video else "",
        ]

        print(" ".join(le_cmd))
        _sproc = subprocess.Popen(
            le_cmd
        )
        while True:
            # TODO: implement model timeout
            _ret_code = _sproc.poll()
            if _ret_code != None:
                break
    else:
        print(f"[ ERROR ] -> Desotacontrolvideo Request Failed: No Input found")
        exit(1)

    if not os.path.isfile(out_filepath):
        print(f"[ ERROR ] -> Desotacontrolvideo Request Failed: No Output found")
        exit(2)
    
    if dev_mode:
        if not report_path.endswith(".json"):
            report_path += ".json"
        with open(report_path, "w") as rw:
            json.dump(
                {
                    "Model Result Path": out_filepath,
                    "Processing Time": time.time() - _report_start_time
                },
                rw,
                indent=2
            )
        detools.user_chown(report_path)
        detools.user_chown(out_filepath)
        print(f"Path to report:\n\t{report_path}")
    

        print(f"[ INFO ] -> DesotaControlVideo Made it!")

        # DeSOTA API Response Preparation
        files = []
        with open(out_filepath, 'rb') as fr:
            files.append(('upload[]', fr))
            # DeSOTA API Response Post
            send_task = s.post(url = send_task_url, files=files)
            print(f"[ INFO ] -> DeSOTA API Upload:{json.dumps(send_task.json(), indent=2)}")
        # Delete temporary file
        os.remove(out_filepath)

        if send_task.status_code != 200:
            print(f"[ ERROR ] -> Desotacontrolvideo Post Failed (Info):\nfiles: {files}\nResponse Code: {send_task.status_code}")
            exit(3)
    
    print("TASK OK!")
    exit(0)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.model_req or not args.model_res_url:
        raise EnvironmentError()
    main(args)