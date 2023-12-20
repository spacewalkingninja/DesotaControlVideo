import os
import sys
import shutil
import time
import json
import argparse
import random
import subprocess
# Define the output path for the synthesized video

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(CURRENT_PATH, f"tmp")
OUT_PATH = os.path.join(TMP_PATH, f"out.mp4")
OUT_NAME = "out"
DEF_DATA = os.path.join(CURRENT_PATH, f"data")
DEF_MOON = os.path.join(DEF_DATA, f"moonwalk.mp4")
INF_PATH = os.path.join(CURRENT_PATH, f"inference.py")

preseed = int(random.randint(1, 1000000))  # Generate a random seed

# Create an argument parser
parser = argparse.ArgumentParser(description="AI Video With Text Control Synthesis CLI")
parser.add_argument("-p", "--prompt", default="enter_cli_mode", help="Text description of target video", type=str)
parser.add_argument("-vp", "--video_path", default=DEF_MOON, help="Path to the source video file", type=str)
parser.add_argument("-c", "--condition", help="Condition of structure sequence", choices=["depth", "canny", "pose", "scribble_hed", "softedge_hed", "scribble_hedsafe", "softedge_hedsafe", "depth_midas", "mlsd", "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand", "scribble_pidinet", "softedge_pidinet", "scribble_pidsafe", "softedge_pidsafe", "normal_bae", "lineart_coarse", "lineart_realistic", "lineart_anime", "depth_zoe", "depth_leres", "depth_leres++", "shuffle", "mediapipe_face", "canny"], type=str)
parser.add_argument("-vl", "--video_length", help="Length of synthesized video", default=15, type=int)
parser.add_argument("-fr", "--frame_rate", help="FrameRate of SyntVideo, defautls to 2", default=2, type=int)
parser.add_argument("-vw", "--width", help="Output Video Width defaults to 256px", default=256, type=int)
parser.add_argument("-vh", "--height", help="Output Video Height defaults to 256px", default=256, type=int)
#parser.add_argument("-ss", "--smoother_steps", help="Timesteps for using interleaved-frame smoother", default="19 20", type=str)
parser.add_argument("-lv", "--is_long_video", help="Set 1 to produce a long video OR use to lower RAM usage; Set 0 to disable; LongVid is On by default!", default=1, action="store_true")
parser.add_argument("-eo", "--extract_only", help="Set 1 to extract only the condition(eg. pose); Set 0 to proceed to inference; Default 0!", default=0, action="store_true")
parser.add_argument("-vc", "--version_controlnet", help="Controlnet version can be v10 or v11, def10", default="v10", action="store_true")
parser.add_argument("-ni", "--num_inference_steps", help="Number of denoising steps", default=50, type=int)
parser.add_argument("-gs", "--guidance_scale", help="Scale for classifier-free guidance", default=12.5, type=float)
parser.add_argument("-s", "--seed", help="Random seed for generation (leave blank to randomize)", default=preseed, type=int)

args = parser.parse_args()




#parser = argparse.ArgumentParser()
if not args.extract_only:
    DEFAULT_OUT_PATH = os.path.join(f"{args.condition}-edit.mp4")
else:
    DEFAULT_OUT_PATH = os.path.join(f"{args.condition}-only-video.mp4")
parser.add_argument("-rp", "--respath", 
                    help=f'Output mp4 filename, do not change... default `./condition-(edit|only-video).mp4`',
                    default=str(DEFAULT_OUT_PATH),
                    type=str)
parser.add_argument('-nc', '--noclear',
                    help='Service Status Request',
                    action='store_true')

args = parser.parse_args()

if args.is_long_video == 1:
    args.is_long_video = True
if args.is_long_video != 1:
    args.is_long_video = False


args.query = args.prompt

DEBUG = True
    
# UTILS
def pcol(obj, template, nostart=False, noend=False):
    '''
    # Description
        print with colors
    # Arguments
    {
        obj: {
            desc: object to print, parsed into string
        },
        template: {
            desc: template name,
            options: [
                header1,
                header2,
                section,
                title,
                body,
                sucess,
                fail
            ]
        }
    }
    '''
    _configs = {
        "header1": "\033[1;105m",
        "header2": "\033[1;95m",
        "search": "\033[104m",
        "section": "\033[94m",
        "title": "\033[7m",
        "body": "\033[97m",
        "sucess": "\033[92m",
        "fail": "\033[91m",
        "end": "\033[0m"
    }
    _morfed_obj = ""
    # PARSE OBJ INTO STR
    if isinstance(obj, list) or isinstance(obj, dict):
        _morfed_obj = json.dumps(obj, indent=2)
    elif not isinstance(obj, str):
        try:
            _morfed_obj = str(obj)
        except:
            # Last ressource
            pass
    else:
        _morfed_obj = obj

    if template in _configs and (_morfed_obj or _morfed_obj==""):
        return f"{_configs[template] if not nostart else ''}{_morfed_obj}{_configs['end'] if not noend else ''}"
    else:
        return obj

def VideoThread(query, args=args):
    prompt = query
    args = args
    res = None
    # Check that required arguments are provided
    if not args.prompt or not args.video_path or not args.condition:
        sys.exit("Please provide a prompt, video path, and condition")

    # Convert source video path to Path object
    video_path = args.video_path
    output_path = args.respath
    if not os.path.isfile(video_path):
        sys.exit("Source video not found")
    try:
        shutil.rmtree(TMP_PATH)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    # Build the command for subprocess
    command = [
        "python",
        INF_PATH,
        "--prompt", str(args.prompt),
        "--condition", str(args.condition),
        "--video_path", str(video_path),
        "--output_path", str(TMP_PATH),  # You may need to adjust this
        "--temp_video_name", str(DEFAULT_OUT_PATH),  # You may need to adjust this
        "--video_length", str(args.video_length),
#            "--smoother_steps", str(f'"{args.smoother_steps}"'),
        "--width", str(args.width),
        "--height", str(args.height),
        "--frame_rate", str(args.frame_rate),
        "--version",  str(args.version_controlnet),  # You may need to adjust this
        "--seed", str(args.seed) if args.seed.isdigit() else str(random.randint(1, 1000000)),
    ]

    if args.is_long_video:
        command.append("--is_long_video")

    if args.extract_only:
        command.append("--extract_only")

    # Execute the subprocess
    subprocess.run(command, check=True)
    return OUT_PATH

def main(args):
    if args.prompt == "enter_cli_mode":
        if not args.noclear:
            os.system("cls" if sys.platform == "win32" else "clear" )
        print(pcol("Welcome to Desota AI Text Video Control CLI ", "header1"), pcol("by © DeSOTA, 2023", "header2"))
        print(pcol("Edit video with text!\n", "body"))

        while True:
            print(pcol("*"*80, "body"))

            # Get User Query
            _user_query = ""
            _exit = False
            try:
                _input_query_msg = "".join([pcol("What is your text prompt? ('exit' to exit)\n-------------------------------------------\n|", "search"), pcol("", "title", noend=True)])
                _user_query = input(_input_query_msg)
            except KeyboardInterrupt:
                _exit = True
                pass
            if _user_query in ["exit", "Exit", "EXIT"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            #now we have to make this kind of question for every item of args.
            #and at the end display them all
            # and ask you sure you wanna go with these settings? Y/N

            args.prompt = _user_query
            args.query = _user_query



            # Get User Input for -vp/--video_path Argument
            try:
                _user_query = input(f"Enter the path to the source video file (default is {DEF_MOON}): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.video_path = _user_query if _user_query else DEF_MOON

            # Get User Input for -vc/--version_controlnet Argument
            try:
                _user_query = input("Enter the Controlnet version (choices: v10 or v11, default is v10): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.version_controlnet = _user_query.lower() if _user_query else "v10"

            # Get User Input for -c/--condition Argument
            try:
                _user_query = input("Enter the condition of structure sequence (choices: v10: depth, canny, pose; v11: scribble_hed, softedge_hed, scribble_hedsafe, softedge_hedsafe, depth_midas, mlsd, openpose, openpose_face, openpose_faceonly, openpose_full, openpose_hand, scribble_pidinet, softedge_pidinet, scribble_pidsafe, softedge_pidsafe, normal_bae, lineart_coarse, lineart_realistic, lineart_anime, depth_zoe, depth_leres, depth_leres++, shuffle, mediapipe_face, canny): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.condition = _user_query


            # Get User Input for -lv/--is_long_video Argument
            try:
                _user_query = input("Set 1 to extract condition only (for example to extract the pose): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.extract_only = True if _user_query.lower() == "1" else False

            # Get User Input for -vl/--video_length Argument
            try:
                _user_query = input("Enter the length of synthesized video (default is 15): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.video_length = int(_user_query) if _user_query else 15


            # Get User Input for -vl/--video_length Argument
            try:
                _user_query = input("Enter the width; default is 256: ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.width = int(_user_query) if _user_query else 256


            # Get User Input for -vl/--video_length Argument
            try:
                _user_query = input("Enter the height; default is 256: ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.height = int(_user_query) if _user_query else 256


            # Get User Input for -fr/--frame_rate Argument
            try:
                _user_query = input("Enter the frame rate of SyntVideo (default is 2): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.frame_rate = int(_user_query) if _user_query else 2

            # Get User Input for -ss/--smoother_steps Argument
            #try:
            #    _user_query = input("Enter the timesteps for using interleaved-frame smoother (default is '19, 20'): ")
            #except KeyboardInterrupt:
            #    print()
            #    return
            #if _user_query.lower() in ["exit"] or _exit:
            #    print(pcol("", "title", nostart=True))
            #    return
            #print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            #args.smoother_steps = _user_query if _user_query else "19 20"

            # Get User Input for -lv/--is_long_video Argument
            try:
                _user_query = input("Set 1 to produce a long video OR use to lower RAM usage; Set 0 to disable; LongVid is On by default! (default is 1): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.is_long_video = False if _user_query.lower() == "0" else True

            # Get User Input for -ni/--num_inference_steps Argument
            try:
                _user_query = input("Enter the number of denoising steps (default is 50): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.num_inference_steps = int(_user_query) if _user_query else 50

            # Get User Input for -gs/--guidance_scale Argument
            try:
                _user_query = input("Enter the scale for classifier-free guidance (default is 12.5): ")
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.guidance_scale = float(_user_query) if _user_query else 12.5

            # Get User Input for -s/--seed Argument
            try:
                _user_query = input("Do you want to enter a specific seed for randomness? (Y/N): ")
                if _user_query.lower() == "y":
                    _user_query = input("Enter the random seed for generation: ")
                    args.seed = _user_query
                else:
                    args.seed = str(random.randint(1, 1000000))  # Generate a random seed
            except KeyboardInterrupt:
                print()
                return
            if _user_query.lower() in ["exit"] or _exit:
                print(pcol("", "title", nostart=True))
                return
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            args.seed = _user_query

            os.system("cls" if sys.platform == "win32" else "clear" )
            print(pcol("Desota AI Text Video Control CLI ", "header1"), pcol("by © DeSOTA, 2023", "header2"))
            print(pcol("Confirm Settings!\n", "body"))

            # Display the chosen settings
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            print("\nChosen Settings:")
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            print(f"Prompt: {args.prompt}")
            print(f"Video Path: {args.video_path}")
            print(f"Version Controlnet: {args.version_controlnet}")
            print(f"Condition: {args.condition}")
            print(f"Condition Only? {args.extract_only}")
            print(f"Output Video Length: {args.video_length}")
            print(f"Output Video Width px: {args.width}")
            print(f"Output Video Height px: {args.height}")
            print(f"Frame Rate: {args.frame_rate}")
            #print(f"Smoother Steps: {args.smoother_steps}")
            print(f"Is Long Video: {args.is_long_video}")
            print(f"Number of Inference Steps: {args.num_inference_steps}")
            print(f"Guidance Scale: {args.guidance_scale}")
            print(f"Seed: {args.seed}")
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            # Ask user for confirmation
            confirm = input("Are you sure you want to go with these settings? (Y/N): ")
            if confirm.lower() != "y":
                print("Settings not confirmed. Exiting.")
                exit()

            os.system("cls" if sys.platform == "win32" else "clear" )
            print(pcol("Desota AI Text Video Control CLI ", "header1"), pcol("by © DeSOTA, 2023", "header2"))
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            print("\nChosen Settings:")
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            print(f"Prompt: {args.prompt}")
            print(f"Video Path: {args.video_path}")
            print(f"Version Controlnet: {args.version_controlnet}")
            print(f"Condition: {args.condition}")
            print(f"Condition Only? {args.extract_only}")
            print(f"Output Video Length: {args.video_length}")
            print(f"(Width px: {args.width} | Height px: {args.height}) @ {args.frame_rate}FPS")
            #print(f"Smoother Steps: {args.smoother_steps}")
            print(f"Is Long Video: {args.is_long_video}")
            print(f"Inference Steps: {args.num_inference_steps} | Guidance: {args.guidance_scale} | Seed: {args.seed}")
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            print(pcol("AI RENDER INIT!\n", "body"))
            print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            # Get Results

            _start_time = time.time() if DEBUG else 0
            _url_res = False
            _resnum = 0
            tsearch = VideoThread(args.query, args)
            #tsearch.start()
            #tsearch.join(timeout=15)
            #_url_res = tsearch.res
            
            if DEBUG:
                #print(f" [ DEBUG ] - TimeOut: {tsearch.is_alive()}")
                print(f" [ DEBUG ] - elapsed time (secs): {time.time()-_start_time}")

            # Print Results
            if tsearch:
                print('done... :)')
            #    while not os.path.isfile(_url_res):
            #        print("Output video not found")
            #        time.sleep(0.1)
            #    os.rename(_url_res, args.respath)
            #    os.system("cls" if sys.platform == "win32" else "clear" )
            #    print(pcol("Desota AI Text Video Control CLI ", "header1"), pcol("by © DeSOTA, 2023", "header2"))
            #    print(f'{pcol("", "title", nostart=True)}{pcol("-------------------------------------------", "search")}')
            #    print(f"\nSaved at: {args.respath}")
                

    else:
        _url_res = False
        _resnum = 0
        tsearch = VideoThread(args.query, args)
        #tsearch.start()
        #tsearch.join(timeout=15)
        #_url_res = tsearch.res
        
        if tsearch:
            print('done... :)')
            #while not os.path.isfile(_url_res):
            #    print("Output video not found")
            #    time.sleep(0.1)
            #os.rename(tsearch, args.respath)
            
        exit(0)
        
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)