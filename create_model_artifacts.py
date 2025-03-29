import os 
import subprocess

cur_dir = os.getcwd()
onnx_dir_path = os.path.join(cur_dir,"onnx")
tidl_tools_path = os.path.join(cur_dir,"edgeai-tidl-tools/tidl_tools")
config_dir_path = os.path.join(cur_dir,"configs")
print(f"cur dir : {cur_dir}")
print(f"onnx dir path : {onnx_dir_path}")
print(f"tidl tools path : {tidl_tools_path}")
print(f"config dir path : {config_dir_path}")

create_tidl_configs = False
# create_tidl_configs = True
# create_tidl_models  = False
create_tidl_models  = True

#create tidl import configs
if create_tidl_configs:
    for filename in os.listdir(onnx_dir_path):
        onnx_model_path = os.path.join(onnx_dir_path,filename)
        model_name = filename.split('.')[0]
        tidl_model_path = os.path.join(cur_dir,"model_artifacts",model_name)
        os.makedirs(tidl_model_path,exist_ok=True)

        data = {
            "modelType": 2,
            "numFrames": 1,
            "inputNetFile": onnx_model_path,
            "outputNetFile": os.path.join(tidl_model_path,"tidl_net.bin"),
            "outputParamsFile": os.path.join(tidl_model_path,"tidl_io_" ),
            "numParamBits": 8,
            "numFeatureBits": 8,
            "inWidth": 224,
            "inHeight": 224,
            "inNumChannels": 3,
            "inData": os.path.join(cur_dir,"input_data/input.bin"),
            "inFileFormat": 1,
            "rawDataInElementType": 0,

            #Config Params for path of different modules 
            "tidlStatsTool": os.path.join(tidl_tools_path,"PC_dsp_test_dl_algo.out"),
            "perfSimTool": os.path.join(tidl_tools_path,"ti_cnnperfsim.out"),
            "graphVizTool": os.path.join(tidl_tools_path,"tidl_graphVisualiser.out"),

            #Config Params for Graph Compiler
            "perfSimConfig": os.path.join(tidl_tools_path,"device_config.cfg"),

            #Config Params for format conversion
            "inLayout": 0,
            "outElementType": 0,
            "outLayout": 0,

        }

        output_file = os.path.join(cur_dir, "configs/tidl_import_" + model_name + ".txt")
        print(f"Config file path : {output_file}")

        with open(output_file, "w") as f:
            for key, value in data.items():
                f.write(f"{key} = {value}\n")

        print(f"Config Created and Saved to {output_file}")


#create tidl models
if create_tidl_models:
    print("\nCreating TIDL Models...")
    print("============================")
    for filename in os.listdir(config_dir_path):
        config_path = os.path.join(config_dir_path, filename)
        app_path    = os.path.join(tidl_tools_path,"tidl_model_import.out")
        print(f"Processing Config File : {config_path}")
        command = [app_path, config_path]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"TIDL import Completed for {config_path}.")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running application with {config_path}: {e}")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
        except FileNotFoundError:
            print(f"Error: Application not found at '{app_path}'")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        


